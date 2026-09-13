import { createRequire } from "node:module";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import process from "node:process";

const repositoryRoot = resolve(import.meta.dirname, "../..");
const require = createRequire(resolve(repositoryRoot, "web/package.json"));
const { chromium } = require("@playwright/test");

const settingsPath = process.argv[2];
const outputPath = process.argv[3];
const mode = process.argv[4] ?? "decode";
if (settingsPath === undefined || outputPath === undefined) {
  throw new Error("usage: webrtc_decode.mjs SETTINGS OUTPUT [decode|denied]");
}

const settings = JSON.parse(await readFile(settingsPath, "utf8"));
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 960, height: 540 } });
const browserErrors = [];
page.on("console", (message) => {
  if (message.type() === "error") browserErrors.push(message.text());
});
page.on("pageerror", (error) => browserErrors.push(error.message));

try {
  await page.goto(settings.origin, { waitUntil: "domcontentloaded" });
  const observation = mode === "denied"
    ? await page.evaluate(async ({ camera_id: cameraId }) => {
        const response = await fetch(`/api/live/${cameraId}/whep`, {
          method: "POST",
          headers: { "Content-Type": "application/sdp" },
          body: "unauthorized-offer",
        });
        return { status_code: response.status };
      }, settings)
    : await page.evaluate(async ({ camera_id: cameraId, qa_session: qaSession }) => {
        const peer = new RTCPeerConnection({ iceServers: [] });
        peer.addTransceiver("video", { direction: "recvonly" });
        const video = document.querySelector("video");
        const trackReady = new Promise((resolveTrack) => {
          peer.ontrack = (event) => {
            video.srcObject = event.streams[0] ?? new MediaStream([event.track]);
            resolveTrack(true);
          };
        });
        const offer = await peer.createOffer();
        await peer.setLocalDescription(offer);
        if (peer.iceGatheringState !== "complete") {
          await new Promise((resolveIce) => {
            peer.addEventListener("icegatheringstatechange", () => {
              if (peer.iceGatheringState === "complete") resolveIce(true);
            });
          });
        }
        const response = await fetch(`/api/live/${cameraId}/whep`, {
          method: "POST",
          headers: {
            "Content-Type": "application/sdp",
            "X-GW-QA-Session": qaSession,
          },
          body: peer.localDescription.sdp,
        });
        if (response.status !== 201) throw new Error(`WHEP POST returned ${response.status}`);
        const resourceLocation = response.headers.get("Location");
        if (resourceLocation === null) throw new Error("WHEP Location missing");
        await peer.setRemoteDescription({ type: "answer", sdp: await response.text() });
        await Promise.race([
          trackReady,
          new Promise((_, reject) => setTimeout(() => reject(new Error("track timeout")), 10000)),
        ]);
        await video.play();
        const samples = [];
        let codecMimeType = null;
        const deadline = performance.now() + 10000;
        while (performance.now() < deadline) {
          const stats = await peer.getStats();
          for (const report of stats.values()) {
            if (report.type === "inbound-rtp" && report.kind === "video") {
              const codec = stats.get(report.codecId);
              if (codec?.type === "codec") codecMimeType = codec.mimeType;
              samples.push({
                frames_decoded: report.framesDecoded ?? 0,
                frames_received: report.framesReceived ?? 0,
                packets_received: report.packetsReceived ?? 0,
              });
            }
          }
          if (samples.length >= 2 && samples.at(-1).frames_decoded > samples[0].frames_decoded) break;
          await new Promise((resolvePoll) => setTimeout(resolvePoll, 100));
        }
        const connectedState = peer.connectionState;
        window.__gwPeer = peer;
        return {
          post_status: response.status,
          delete_status: 0,
          resource_location: resourceLocation,
          connection_state: connectedState,
          transceiver_direction: "recvonly",
          codec_mime_type: codecMimeType,
          samples,
          video_width: video.videoWidth,
          video_height: video.videoHeight,
        };
      }, settings);
  if (mode === "denied" && observation.status_code !== 401) {
    throw new Error(`expected 401, received ${observation.status_code}`);
  }
  if (mode === "decode") {
    const samples = observation.samples;
    if (samples.length < 2 || samples.at(-1).frames_decoded <= samples[0].frames_decoded) {
      throw new Error("framesDecoded did not rise");
    }
    await page.screenshot({ path: outputPath.replace(/\.json$/, ".png") });
    observation.delete_status = await page.evaluate(
      async ({ resourceLocation, qaSession }) => {
        const response = await fetch(resourceLocation, {
          method: "DELETE",
          headers: { "X-GW-QA-Session": qaSession },
        });
        window.__gwPeer.close();
        return response.status;
      },
      { resourceLocation: observation.resource_location, qaSession: settings.qa_session },
    );
    if (observation.delete_status < 200 || observation.delete_status >= 300) {
      throw new Error(`WHEP DELETE returned ${observation.delete_status}`);
    }
  }
  await writeFile(
    outputPath,
    `${JSON.stringify({ mode, observation, browser_errors: browserErrors }, null, 2)}\n`,
  );
} finally {
  await browser.close();
}
