// Create pyengine/.venv and install the engine's Python packages into it.
// Set MLSIM_PYTHON to choose the base interpreter (Python 3.9 or newer).
const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const engineDir = path.join(__dirname, "..", "pyengine");
const venvDir = path.join(engineDir, ".venv");
const venvPython =
  process.platform === "win32"
    ? path.join(venvDir, "Scripts", "python.exe")
    : path.join(venvDir, "bin", "python");

function run(command, args) {
  console.log(">", command, args.join(" "));
  const result = spawnSync(command, args, { stdio: "inherit" });
  if (result.error || result.status !== 0) {
    console.error(
      result.error ? result.error.message : `${command} exited with ${result.status}`
    );
    process.exit(1);
  }
}

if (!fs.existsSync(venvPython)) {
  const base =
    process.env.MLSIM_PYTHON || (process.platform === "win32" ? "python" : "python3");
  run(base, ["-m", "venv", venvDir]);
}
run(venvPython, ["-m", "pip", "install", "--upgrade", "pip"]);
run(venvPython, ["-m", "pip", "install", "-r", path.join(engineDir, "requirements.txt")]);
console.log("\nThe engine environment is ready:", venvPython);
