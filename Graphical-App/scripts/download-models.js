// Download the pre-trained models into pyengine/models.
const path = require("path");
const { ensureModels } = require("../electron/models");

const dir = path.join(__dirname, "..", "pyengine", "models");
let last = -1;

ensureModels(dir, (fraction) => {
  const percent = Math.floor(fraction * 100);
  if (percent !== last) {
    last = percent;
    process.stdout.write(`Downloading: ${percent} %\r`);
  }
})
  .then(() => console.log("\nModels are ready in", dir))
  .catch((err) => {
    console.error("\n" + err.message);
    process.exit(1);
  });
