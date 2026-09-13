import { createRequire } from "node:module";
import { access, chmod, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const repositoryRoot = resolve(import.meta.dirname, "../../../../../");
const require = createRequire(resolve(repositoryRoot, "web/package.json"));
const { chromium } = require("@playwright/test");

const [origin, passwordFile, cameraId, outputPath, mode, profileA, profileB] = process.argv.slice(2);
if ([origin, passwordFile, cameraId, outputPath, mode, profileA, profileB].some((value) => value === undefined)) {
  throw new Error("Task 12 live browser arguments are incomplete");
}

const root = dirname(outputPath);
const password = (await readFile(passwordFile, "utf8")).trim();
const controlPort = Number(process.env.GW_TASK12_CONTROL_PORT);
const controlUser = process.env.GW_TASK12_CONTROL_USER;
const controlPassword = process.env.GW_TASK12_CONTROL_PASSWORD;
const mediaPath = process.env.GW_TASK12_MEDIA_PATH;
const applicationOrigin = new URL(origin).origin;
const errors = [];

function attach(page) {
  page.on("pageerror", (error) => errors.push(error.message));
}

function trackApplicationRequests(page) {
  let armed = false;
  let count = 0;
  page.on("request", (request) => {
    if (!armed) return;
    const url = new URL(request.url());
    if (url.origin === applicationOrigin && url.pathname.startsWith("/api/")) count += 1;
  });
  return {
    arm() {
      armed = true;
    },
    count() {
      return count;
    },
  };
}

async function request(page, path, options = {}) {
  return page.evaluate(async ({ path, options }) => {
    const response = await fetch(path, options);
    const text = await response.text();
    let body = null;
    try {
      body = JSON.parse(text);
    } catch {
      body = text.length === 0 ? null : text;
    }
    return {
      status: response.status,
      location: response.headers.get("Location"),
      retry_after: response.headers.get("Retry-After"),
      body,
    };
  }, { path, options });
}

async function readerCount() {
  const authorization = Buffer.from(`${controlUser}:${controlPassword}`).toString("base64");
  const response = await fetch(`http://127.0.0.1:${controlPort}/v3/paths/get/${mediaPath}`, {
    headers: { Authorization: `Basic ${authorization}` },
  });
  if (!response.ok) return 0;
  const body = await response.json();
  return Array.isArray(body.readers) ? body.readers.length : 0;
}

async function waitReaders(expected, timeout = 10000, pollInterval = 100) {
  const deadline = Date.now() + timeout;
  let count = await readerCount();
  while (Date.now() < deadline && count !== expected) {
    await new Promise((resolveWait) => setTimeout(resolveWait, pollInterval));
    count = await readerCount();
  }
  return count;
}

async function waitFile(path, timeout = 120000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    try {
      await access(path);
      return;
    } catch {
      await new Promise((resolveWait) => setTimeout(resolveWait, 50));
    }
  }
  throw new Error(`timed out waiting for ${path}`);
}

async function launch(profile) {
  return chromium.launchPersistentContext(profile, {
    args: ["--enable-features=UseSystemCAs", "--use-system-ca-certs", "--disable-dev-shm-usage"],
    headless: true,
    ignoreHTTPSErrors: false,
  });
}

async function login(context, loginPassword = password) {
  const page = await context.newPage();
  attach(page);
  await page.goto(`${origin}/api/session`, { waitUntil: "domcontentloaded" });
  const response = await request(page, "/api/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ password: loginPassword }),
  });
  if (response.status !== 200) throw new Error(`live browser login failed: ${response.status}`);
  return page;
}

async function saveCookie(context, name) {
  const cookie = (await context.cookies(origin)).find((item) => item.name === "gw_session");
  if (cookie === undefined) throw new Error("live browser session cookie missing");
  const path = `${root}/secrets/${name}.cookie`;
  const netscape = [
    "# Netscape HTTP Cookie File",
    `127.0.0.1\tFALSE\t${cookie.path}\t${cookie.secure ? "TRUE" : "FALSE"}\t0\t${cookie.name}\t${cookie.value}`,
    "",
  ].join("\n");
  await writeFile(path, netscape, { mode: 0o600 });
  await chmod(path, 0o600);
  return path;
}

async function startReader(page, trickle) {
  return page.evaluate(async ({ cameraId, trickle }) => {
    const peer = new RTCPeerConnection({ iceServers: [] });
    peer.addTransceiver("video", { direction: "recvonly" });
    const video = document.createElement("video");
    video.autoplay = true;
    video.muted = true;
    document.body.append(video);
    const candidates = [];
    const track = new Promise((resolveTrack) => {
      peer.ontrack = (event) => {
        video.srcObject = event.streams[0] ?? new MediaStream([event.track]);
        resolveTrack();
      };
    });
    peer.onicecandidate = (event) => {
      if (event.candidate !== null) candidates.push(event.candidate);
    };
    const offer = await peer.createOffer();
    await peer.setLocalDescription(offer);
    if (!trickle && peer.iceGatheringState !== "complete") {
      await new Promise((resolveIce) => {
        const finishIce = () => {
          if (peer.iceGatheringState !== "complete") return;
          peer.removeEventListener("icegatheringstatechange", finishIce);
          resolveIce();
        };
        peer.addEventListener("icegatheringstatechange", finishIce);
        finishIce();
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
      return { post_status: response.status, location, frames_decoded: 0, first_frames_decoded: 0, codec: null, fragment: null, offer: peer.localDescription?.sdp ?? "" };
    }
    const answer = await response.text();
    await peer.setRemoteDescription({ type: "answer", sdp: answer });
    if (trickle) {
      const candidateDeadline = performance.now() + 5000;
      while (candidates.length === 0 && peer.iceGatheringState !== "complete" && performance.now() < candidateDeadline) {
        await new Promise((resolveWait) => setTimeout(resolveWait, 25));
      }
    } else {
      await Promise.race([
        track,
        new Promise((_, reject) => setTimeout(() => reject(new Error("live WHEP track timeout")), 10000)),
      ]);
      await video.play();
    }
    const statsSample = async () => {
      const stats = await peer.getStats();
      let frames = 0;
      let codec = null;
      for (const report of stats.values()) {
        if (report.type === "inbound-rtp" && report.kind === "video") {
          frames = report.framesDecoded ?? 0;
          const codecReport = stats.get(report.codecId);
          if (codecReport?.type === "codec") codec = codecReport.mimeType;
        }
      }
      return { frames, codec };
    };
    const first = await statsSample();
    let current = first;
    if (!trickle) {
      const deadline = performance.now() + 10000;
      while (performance.now() < deadline && current.frames <= first.frames) {
        await new Promise((resolvePoll) => setTimeout(resolvePoll, 100));
        current = await statsSample();
      }
    }
    const localSdp = peer.localDescription?.sdp ?? "";
    const ufrag = localSdp.match(/a=ice-ufrag:([^\r\n]+)/)?.[1] ?? "";
    const pwd = localSdp.match(/a=ice-pwd:([^\r\n]+)/)?.[1] ?? "";
    const mLine = localSdp.match(/m=video[^\r\n]*/)?.[0] ?? "m=video 9 UDP/TLS/RTP/SAVPF 96";
    const mid = localSdp.match(/a=mid:([^\r\n]+)/)?.[1] ?? "0";
    const candidate = candidates[0]?.candidate ?? null;
    const fragment = `${`a=ice-ufrag:${ufrag}`}\r\n${`a=ice-pwd:${pwd}`}\r\n${mLine}\r\n${`a=mid:${mid}`}\r\n${candidate === null ? "a=end-of-candidates" : `a=${candidate}`}\r\n`;
    window.__gwBoundary = { peer, location, offer: localSdp, fragment, track, video };
    return {
      post_status: response.status,
      location,
      frames_decoded: current.frames,
      first_frames_decoded: first.frames,
      codec: current.codec,
      fragment,
      offer: localSdp,
    };
  }, { cameraId, trickle });
}

async function writeResult(result) {
  await writeFile(outputPath, `${JSON.stringify({ ...result, browser_errors: errors }, null, 2)}\n`);
}

async function patchPhase() {
  const first = await launch(profileA);
  const second = await launch(profileB);
  try {
    const pageA = await login(first);
    const reader = await startReader(pageA, true);
    const readersDuring = await waitReaders(1);
    const anonymousPage = await second.newPage();
    attach(anonymousPage);
    await anonymousPage.goto(`${origin}/api/session`, { waitUntil: "domcontentloaded" });
    const anonymous = await request(anonymousPage, reader.location, {
      method: "PATCH",
      headers: { "Content-Type": "application/trickle-ice-sdpfrag" },
      body: reader.fragment,
    });
    const pageB = await login(second);
    const crossSession = await request(pageB, reader.location, {
      method: "PATCH",
      headers: { "Content-Type": "application/trickle-ice-sdpfrag" },
      body: reader.fragment,
    });
    const framesBeforePatch = await frameCount(pageA);
    const own = await request(pageA, reader.location, {
      method: "PATCH",
      headers: { "Content-Type": "application/trickle-ice-sdpfrag" },
      body: reader.fragment,
    });
    const afterPatch = await pageA.evaluate(async (baseline) => {
      const boundary = window.__gwBoundary;
      if (boundary === undefined) return 0;
      await Promise.race([
        boundary.track,
        new Promise((_, reject) => setTimeout(() => reject(new Error("live WHEP track timeout after PATCH")), 10000)),
      ]);
      await boundary.video.play();
      let frames = 0;
      const deadline = performance.now() + 10000;
      while (performance.now() < deadline) {
        const currentStats = await boundary.peer.getStats();
        for (const report of currentStats.values()) {
          if (report.type === "inbound-rtp" && report.kind === "video") frames = report.framesDecoded ?? 0;
        }
        if (frames > baseline) return frames;
        await new Promise((resolvePoll) => setTimeout(resolvePoll, 100));
      }
      return frames;
    }, framesBeforePatch);
    const deleted = await request(pageA, reader.location, { method: "DELETE" });
    const readersAfterDelete = await waitReaders(0);
    await writeResult({
      mode,
      post_status: reader.post_status,
      codec: reader.codec,
      first_frames_decoded: reader.first_frames_decoded,
      frames_decoded: Math.max(reader.frames_decoded, afterPatch),
      frames_before_patch: framesBeforePatch,
      frames_after_patch: afterPatch,
      frames_advanced_after_patch: afterPatch > framesBeforePatch,
      anonymous_patch: anonymous.status,
      cross_session_patch: crossSession.status,
      own_patch: own.status,
      delete_status: deleted.status,
      readers_during: readersDuring,
      readers_after_delete: readersAfterDelete,
      resource_location: reader.location,
      fragment_line_count: reader.fragment.split("\r\n").length,
    });
  } finally {
    await first.close();
    await second.close();
  }
}

async function credentialPhase() {
  const first = await launch(profileA);
  const second = await launch(profileB);
  try {
    const pageA = await login(first);
    const pageB = await login(second);
    const readerA = await startReader(pageA, false);
    const readerB = await startReader(pageB, false);
    const cookieA = await saveCookie(first, "credential-old-a");
    const cookieB = await saveCookie(second, "credential-old-b");
    const readersBefore = await waitReaders(2);
    await writeFile(`${root}/boundary-credential-replacement-ready.json`, `${JSON.stringify({ cookieA, cookieB, readers: readersBefore })}\n`);
    await waitFile(`${root}/boundary-credential-replacement-go.json`);
    const readersAfterReplacement = await waitReaders(0);
    await first.close();
    await second.close();
    await waitFile(`${root}/boundary-credential-noop-go.json`);
    const third = await launch(profileA);
    try {
      const replacementPassword = (await readFile(passwordFile, "utf8")).trim();
      const pageC = await login(third, replacementPassword);
      const readerC = await startReader(pageC, false);
      const cookieC = await saveCookie(third, "credential-new");
      const readersBeforeNoop = await waitReaders(1);
      await writeFile(`${root}/boundary-credential-noop-ready.json`, `${JSON.stringify({ cookieC, readers: readersBeforeNoop, frames: readerC.frames_decoded })}\n`);
      await waitFile(`${root}/boundary-credential-finish-go.json`);
      const readersAfterNoop = await readerCount();
      await pageC.evaluate(() => window.__gwBoundary?.peer?.close());
      await writeResult({
        mode,
        readers_before_replacement: readersBefore,
        readers_after_replacement: readersAfterReplacement,
        readers_before_noop: readersBeforeNoop,
        readers_after_noop: readersAfterNoop,
        old_cookie_files: [cookieA, cookieB],
        new_cookie_file: cookieC,
        old_frames: [readerA.frames_decoded, readerB.frames_decoded],
        new_frames: readerC.frames_decoded,
      });
    } finally {
      await third.close();
    }
  } finally {
    if (first) await first.close().catch(() => {});
    if (second) await second.close().catch(() => {});
  }
}

async function deletePhase() {
  const context = await launch(profileA);
  try {
    const page = await login(context);
    const reader = await startReader(page, false);
    const cookie = await saveCookie(context, "delete-cookie");
    const readers = await waitReaders(1);
    await writeFile(`${root}/boundary-delete-ready.json`, `${JSON.stringify({ cookie, readers, resource: reader.location })}\n`);
    await waitFile(`${root}/boundary-delete-go.json`);
    const readersAfterDelete = await waitReaders(0);
    await page.evaluate(() => window.__gwBoundary.peer.close());
    const cameras = await request(page, "/api/cameras");
    const resurrection = await request(page, `/api/live/${cameraId}/whep`, {
      method: "POST",
      headers: { "Content-Type": "application/sdp" },
      body: reader.offer,
    });
    const lateDelete = await request(page, reader.location, { method: "DELETE" });
    await writeResult({
      mode,
      post_status: reader.post_status,
      codec: reader.codec,
      frames_decoded: reader.frames_decoded,
      readers_during: readers,
      readers_after_delete: readersAfterDelete,
      cameras_status: cameras.status,
      cameras_after_delete: Array.isArray(cameras.body) ? cameras.body.length : -1,
      resurrection_status: resurrection.status,
      late_delete_status: lateDelete.status,
    });
  } finally {
    await context.close();
  }
}

async function waitStableFrames(page) {
  let framesAfterZero = await frameCount(page);
  let stableRounds = 0;
  const stableDeadline = performance.now() + 10000;
  while (performance.now() < stableDeadline && stableRounds < 4) {
    await new Promise((resolveWait) => setTimeout(resolveWait, 250));
    const currentFrames = await frameCount(page);
    if (currentFrames === framesAfterZero) {
      stableRounds += 1;
    } else {
      stableRounds = 0;
      framesAfterZero = currentFrames;
    }
  }
  return { frames_after_zero: framesAfterZero, frames_stopped: stableRounds >= 4 };
}

async function expiryCase(context, cookieName, columnName) {
  const page = await login(context);
  const requests = trackApplicationRequests(page);
  const reader = await startReader(page, false);
  const cookie = await saveCookie(context, cookieName);
  const readers = await waitReaders(1);
  if (readers !== 1) throw new Error(`expiry readers did not start: ${readers}`);
  const framesAtReady = await frameCount(page);
  await writeFile(`${root}/boundary-expiry-${columnName}-ready.json`, `${JSON.stringify({ cookie, readers, frames: framesAtReady })}\n`);
  requests.arm();
  await waitFile(`${root}/boundary-expiry-${columnName}-go.json`);
  const expiredAt = Date.now();
  const readersAfterExpiry = await waitReaders(0, 5000, 25);
  const elapsedMs = Date.now() - expiredAt;
  const stable = await waitStableFrames(page);
  const appRequestsAfterReady = requests.count();
  const zero = {
    readers_zero: readersAfterExpiry,
    [`${columnName}_elapsed_ms`]: elapsedMs,
    frames_at_ready: [framesAtReady],
    frames_after_zero: [stable.frames_after_zero],
    frames_stopped: stable.frames_stopped,
    app_requests_after_ready: appRequestsAfterReady,
    no_app_requests: appRequestsAfterReady === 0,
  };
  await writeFile(`${root}/boundary-expiry-${columnName}-zero.json`, `${JSON.stringify(zero)}\n`);
  await waitFile(`${root}/boundary-expiry-${columnName}-verified.json`);
  await page.evaluate(() => window.__gwBoundary?.peer?.close());
  await page.close();
  return { cookie, reader, readers, zero };
}

async function expiryPhase() {
  const first = await launch(profileA);
  const second = await launch(profileB);
  try {
    const idle = await expiryCase(first, "expiry-idle", "idle");
    const absolute = await expiryCase(second, "expiry-absolute", "absolute");
    await writeResult({
      mode,
      readers_before_expiry: [idle.readers, absolute.readers],
      frames_before_expiry: [idle.reader.frames_decoded, absolute.reader.frames_decoded],
      old_cookie_files: [idle.cookie, absolute.cookie],
      idle: idle.zero,
      absolute: absolute.zero,
    });
  } finally {
    await first.close();
    await second.close();
  }
}

async function frameCount(page) {
  return page.evaluate(async () => {
    const peer = window.__gwBoundary?.peer;
    if (peer === undefined) return 0;
    const stats = await peer.getStats();
    for (const report of stats.values()) {
      if (report.type === "inbound-rtp" && report.kind === "video") return report.framesDecoded ?? 0;
    }
    return 0;
  });
}

if (mode === "patch") await patchPhase();
else if (mode === "credential") await credentialPhase();
else if (mode === "delete") await deletePhase();
else if (mode === "expiry") await expiryPhase();
else throw new Error(`unknown live browser mode: ${mode}`);
