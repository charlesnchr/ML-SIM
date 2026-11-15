// Downloads the pre-trained models and verifies their checksums.
// Used by the Electron main process and by `npm run download-models`.
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { Readable, Transform } = require("stream");
const { pipeline } = require("stream/promises");

const MODELS = require("./models.json");

function sha256(file) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    fs.createReadStream(file)
      .on("error", reject)
      .on("data", (chunk) => hash.update(chunk))
      .on("end", () => resolve(hash.digest("hex")));
  });
}

async function isValid(file, model) {
  return fs.existsSync(file) && (await sha256(file)) === model.sha256;
}

async function download(model, dest, onBytes) {
  const res = await fetch(model.url);
  if (!res.ok) throw new Error(`${model.url} returned HTTP ${res.status}`);
  const size = Number(res.headers.get("content-length")) || 0;
  onBytes(0, size);

  const counter = new Transform({
    transform(chunk, _encoding, callback) {
      onBytes(chunk.length, 0);
      callback(null, chunk);
    },
  });
  const part = dest + ".part";
  await pipeline(Readable.fromWeb(res.body), counter, fs.createWriteStream(part));

  if ((await sha256(part)) !== model.sha256) {
    fs.unlinkSync(part);
    throw new Error(`${model.name} does not match its checksum`);
  }
  fs.renameSync(part, dest);
}

// Make sure every model in models.json is present in `dir`.
// `onProgress(fraction)` is called while files download.
async function ensureModels(dir, onProgress = () => {}, log = console) {
  fs.mkdirSync(dir, { recursive: true });

  const missing = [];
  for (const model of MODELS) {
    if (await isValid(path.join(dir, model.name), model)) {
      log.info(model.name, "is present");
    } else {
      missing.push(model);
    }
  }

  let done = 0;
  let total = 0;
  for (const model of missing) {
    log.info("Downloading", model.url);
    await download(model, path.join(dir, model.name), (bytes, size) => {
      done += bytes;
      total += size;
      if (total > 0) onProgress(Math.min(done / total, 1));
    });
    log.info("Downloaded", model.name);
  }
}

module.exports = { MODELS, ensureModels };
