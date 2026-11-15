const { app, ipcMain } = require("electron");
const fs = require("fs");
const path = require("path");
const log = require("electron-log/main");
const Store = require("electron-store");

const { ensureModels } = require("./models");
const engineCommunication = require("./EngineCommunication.js");

const store = new Store();

/*************************************************************
 * Start the Python engine
 *
 * The engine is the Python source in pyengine/pysrc. It runs in a Python
 * environment that has the packages from pyengine/requirements.txt, which
 * `npm run setup-engine` creates in pyengine/.venv.
 *************************************************************/

let downloadFraction = null; // set while models are downloading
let failure = null; // message shown in the app when the engine cannot start

const engineRoot = () =>
  app.isPackaged
    ? path.join(process.resourcesPath, "pyengine")
    : path.join(app.getAppPath(), "pyengine");

// Packaged apps cannot write next to their resources.
const modelDir = () =>
  app.isPackaged
    ? path.join(app.getPath("userData"), "models")
    : path.join(engineRoot(), "models");

// The interpreter is, in order: $MLSIM_PYTHON, the "pythonPath" setting,
// the virtual environment made by `npm run setup-engine`, the system Python.
const findPython = () => {
  const venvPython =
    process.platform === "win32"
      ? path.join(".venv", "Scripts", "python.exe")
      : path.join(".venv", "bin", "python");
  const candidates = [
    process.env.MLSIM_PYTHON,
    store.get("pythonPath"),
    path.join(engineRoot(), venvPython),
    path.join(app.getPath("userData"), venvPython),
  ];
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }
  return process.platform === "win32" ? "python" : "python3";
};

const initEngine = async () => {
  // The renderer polls this until the engine is active.
  ipcMain.on("EngineStatus", (event) => {
    if (engineCommunication.get_serverActive()) {
      event.sender.send("EngineStatus", "a");
    } else if (failure) {
      event.sender.send("EngineStatus", "x," + failure);
    } else if (downloadFraction != null) {
      event.sender.send(
        "EngineStatus",
        "d," + Math.floor(100 * downloadFraction)
      );
    } else {
      event.sender.send("EngineStatus", "w"); // something else, just wait
    }
  });

  try {
    await ensureModels(
      modelDir(),
      (fraction) => {
        downloadFraction = fraction;
      },
      log
    );
  } catch (err) {
    log.error("Model download failed", err);
    failure = "Could not download the models. " + err.message;
    return;
  } finally {
    downloadFraction = null;
  }

  engineCommunication.createPyProc({
    python: findPython(),
    script: path.join(engineRoot(), "pysrc", "engine.py"),
    modelDir: modelDir(),
    onFailure: (message) => {
      failure = message;
    },
  });
};

module.exports = {
  initEngine,
};
