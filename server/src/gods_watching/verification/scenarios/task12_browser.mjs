import { createRequire } from "node:module";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const repositoryRoot = resolve(import.meta.dirname, "../../../../../");
const require = createRequire(resolve(repositoryRoot, "web/package.json"));
const { chromium } = require("@playwright/test");

const origin = process.argv[2];
const password = process.argv[3];
const cameraId = process.argv[4];
const outputPath = process.argv[5];
const mode = process.argv[6] ?? "happy";
const profilePath = process.argv[7];
if ([origin, password, cameraId, outputPath, profilePath].some((value) => value === undefined)) {
  throw new Error("Task 12 browser arguments are incomplete");
}

const context = await chromium.launchPersistentContext(profilePath, {
  args: [
    "--enable-features=UseSystemCAs",
    "--use-system-ca-certs",
    "--disable-dev-shm-usage",
  ],
  headless: true,
  ignoreHTTPSErrors: false,
});
const contexts = [context];
const errors = [];

function attach(page) {
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
}

async function request(page, path, options = {}) {
  return page.evaluate(async ({ path, options }) => {
    const response = await fetch(path, options);
    const text = await response.text();
    let body = null;
    try {
      body = JSON.parse(text);
    } catch {
      body = text.length === 0 ? null : "non_json_body";
    }
    return {
      status: response.status,
      retry_after: response.headers.get("Retry-After"),
      location: response.headers.get("Location"),
      body,
    };
  }, { path, options });
}

async function readerCount() {
  const port = Number(process.env.GW_TASK12_CONTROL_PORT);
  const user = process.env.GW_TASK12_CONTROL_USER;
  const secret = process.env.GW_TASK12_CONTROL_PASSWORD;
  const path = process.env.GW_TASK12_MEDIA_PATH;
  const authorization = Buffer.from(`${user}:${secret}`).toString("base64");
  const response = await fetch(`http://127.0.0.1:${port}/v3/paths/get/${path}`, {
    headers: { Authorization: `Basic ${authorization}` },
  });
  const body = await response.json();
  return Array.isArray(body.readers) ? body.readers.length : 0;
}

async function waitReaders(expected) {
  const deadline = Date.now() + 8000;
  let count = await readerCount();
  while (Date.now() < deadline && count !== expected) {
    await new Promise((resolveWait) => setTimeout(resolveWait, 200));
    count = await readerCount();
  }
  return count;
}

async function openPage(context) {
  const page = await context.newPage();
  attach(page);
  await page.goto(`${origin}/api/session`, { waitUntil: "domcontentloaded" });
  return page;
}

async function cookieSummary(context) {
  const cookie = (await context.cookies(origin)).find((item) => item.name === "gw_session");
  if (cookie === undefined) return null;
  return {
    secure: cookie.secure,
    http_only: cookie.httpOnly,
    same_site: cookie.sameSite,
    value_length: cookie.value.length,
  };
}

async function whep(page) {
  return page.evaluate(async ({ cameraId }) => {
    const peer = new RTCPeerConnection({ iceServers: [] });
    peer.addTransceiver("video", { direction: "recvonly" });
    const video = document.createElement("video");
    video.autoplay = true;
    video.muted = true;
    document.body.append(video);
    const track = new Promise((resolveTrack) => {
      peer.ontrack = (event) => {
        video.srcObject = event.streams[0] ?? new MediaStream([event.track]);
        resolveTrack();
      };
    });
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    if (peer.iceGatheringState !== "complete") {
      await new Promise((resolveIce) => {
        peer.addEventListener("icegatheringstatechange", () => {
          if (peer.iceGatheringState === "complete") resolveIce();
        });
      });
    }
    const response = await fetch(`/api/live/${cameraId}/whep`, {
      method: "POST",
      headers: { "Content-Type": "application/sdp" },
      body: peer.localDescription.sdp,
    });
    const location = response.headers.get("Location");
    if (response.status !== 201 || location === null) {
      peer.close();
      return { post_status: response.status, location, frames_decoded: 0, codec: null };
    }
    await peer.setRemoteDescription({ type: "answer", sdp: await response.text() });
    await Promise.race([
      track,
      new Promise((_, reject) => setTimeout(() => reject(new Error("WHEP track timeout")), 10000)),
    ]);
    await video.play();
    const samples = [];
    let codec = null;
    const deadline = performance.now() + 10000;
    while (performance.now() < deadline) {
      const stats = await peer.getStats();
      for (const report of stats.values()) {
        if (report.type === "inbound-rtp" && report.kind === "video") {
          const codecReport = stats.get(report.codecId);
          if (codecReport?.type === "codec") codec = codecReport.mimeType;
          samples.push(report.framesDecoded ?? 0);
        }
      }
      if (samples.length > 1 && samples.at(-1) > samples[0]) break;
      await new Promise((resolvePoll) => setTimeout(resolvePoll, 100));
    }
    window.__gwPeer = peer;
    return {
      post_status: response.status,
      location,
      frames_decoded: samples.at(-1) ?? 0,
      first_frames_decoded: samples[0] ?? 0,
      codec,
      connection_state: peer.connectionState,
    };
  }, { cameraId });
}

async function happy() {
  const page = await openPage(context);
  const anonymous = await request(page, "/api/cameras");
  const anonymousWhep = await request(page, `/api/live/${cameraId}/whep`, {
    method: "POST",
    headers: { "Content-Type": "application/sdp" },
    body: "unauthorized-offer",
  });
  const login = await request(page, "/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  });
  const sessionBefore = await request(page, "/api/session");
  const cameras = await request(page, "/api/cameras");
  const cameraText = JSON.stringify(cameras.body);
  const stream = await whep(page);
  const readersDuring = await waitReaders(1);
  const sessionAfter = await request(page, "/api/session");
  const cookie = await cookieSummary(context);
  const logout = await request(page, "/api/session", { method: "DELETE" });
  const readersAfter = await waitReaders(0);
  const afterLogout = await request(page, "/api/session");
  await page.evaluate(() => window.__gwPeer?.close());
  return {
    anonymous_list: anonymous.status,
    anonymous_whep: anonymousWhep.status,
    login: login.status,
    cookie,
    camera_response: cameras.status,
    source_hidden: !cameraText.toLowerCase().includes("rtsp://"),
    stream,
    readers_during: readersDuring,
    passive_idle_unchanged: sessionBefore.body?.idle_expires_at === sessionAfter.body?.idle_expires_at,
    logout: logout.status,
    readers_after_logout: readersAfter,
    after_logout_authenticated: afterLogout.body?.authenticated === true,
  };
}

async function denied() {
  const page = await openPage(context);
  const anonymousList = await request(page, "/api/cameras");
  const anonymousWhep = await request(page, `/api/live/${cameraId}/whep`, {
    method: "POST",
    headers: { "Content-Type": "application/sdp" },
    body: "unauthorized-offer",
  });
  const login = await request(page, "/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password }),
  });
  const stillAuthenticated = await request(page, "/api/session");
  const malformed = await request(page, "/api/cameras/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_url: "http://not-rtsp.example/source" }),
  });
  const passiveBefore = await request(page, "/api/session");
  const stream = await whep(page);
  const passiveAfter = await request(page, "/api/session");
  const logout = await request(page, "/api/session", { method: "DELETE" });
  await page.evaluate(() => window.__gwPeer?.close());
  const throttlePage = await openPage(context);
  const failures = [];
  for (let index = 0; index < 6; index += 1) {
    failures.push(await request(throttlePage, "/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: `${password}-wrong` }),
    }));
  }
  return {
    anonymous_list: anonymousList.status,
    anonymous_whep: anonymousWhep.status,
    wrong_login_statuses: failures.map((item) => item.status),
    last_retry_after: failures.at(-1)?.retry_after,
    cross_origin_login: null,
    login: login.status,
    cross_origin_logout: null,
    survives_cross_origin: stillAuthenticated.body?.authenticated === true,
    malformed_source: malformed.status,
    passive_idle_unchanged: passiveBefore.body?.idle_expires_at === passiveAfter.body?.idle_expires_at,
    stream,
    logout: logout.status,
  };
}

const result = {
  mode,
  ...(mode === "denied" ? await denied() : await happy()),
  browser_errors: errors,
};
await writeFile(outputPath, `${JSON.stringify(result, null, 2)}\n`);
for (const context of contexts) await context.close();
