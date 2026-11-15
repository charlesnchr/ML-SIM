const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const crypto = require("crypto");

const log = require("electron-log/main");
const Store = require("electron-store");
const remoteMain = require("@electron/remote/main");

log.initialize();
remoteMain.initialize();
Store.initRenderer();

const store = new Store();
const isDev = !app.isPackaged;

// The Vite dev server, when the app is started with `npm start`.
const devServerUrl = process.env.MLSIM_DEV_SERVER;
const indexHtml = path.join(__dirname, "..", "build", "index.html");

let mainWindow;

let exportdir = store.get("exportdir");
if (exportdir == null) {
  exportdir = path.join(app.getPath("documents"), "Mambio");
  store.set("exportdir", exportdir);
}
let cachedir = store.get("cachedir");
if (cachedir == null) {
  cachedir = path.join(app.getPath("appData"), "Mambio-Library", "2");
  store.set("cachedir", cachedir);
}

global.settings = {
  handler: null,
  cachedir: cachedir,
  exportdir: exportdir,
};

const engineSetup = require("./EngineSetup.js");
const engineCommunication = require("./EngineCommunication.js");

app.on("ready", engineSetup.initEngine);
app.on("will-quit", engineCommunication.exitEngine);

/*************************************************************
 * window management
 *************************************************************/

function createAppWindow(route) {
  const win = new BrowserWindow({
    width: 1000,
    height: 700,
    minWidth: 700,
    minHeight: 500,
    title: "ML-SIM",
    backgroundColor: "#444444",
    // The custom header doubles as the title bar on macOS only.
    titleBarStyle: process.platform === "darwin" ? "hidden" : "default",
    webPreferences: {
      devTools: isDev,
      // The renderer reads folders and talks to the main process directly.
      nodeIntegration: true,
      contextIsolation: false,
      sandbox: false,
      // Thumbnails and results are loaded from disk with file:// URLs.
      webSecurity: false,
    },
  });
  remoteMain.enable(win.webContents);

  if (devServerUrl) {
    win.loadURL(devServerUrl + (route ? "#" + route : ""));
  } else {
    win.loadFile(indexHtml, route ? { hash: route } : undefined);
  }
  return win;
}

ipcMain.on("showImage", (event, imgsrc) => {
  if (!imgsrc) return;
  const imageWindow = createAppWindow("/imageview");
  imageWindow.setMenuBarVisibility(false);
  imageWindow.webContents.on("dom-ready", () => {
    imageWindow.webContents.send(
      "filepathArgument",
      imgsrc.replace(/\\/g, "/")
    );
  });
});

ipcMain.on("open-file-dialog", (event) => {
  dialog
    .showOpenDialog({
      properties: ["openDirectory"],
    })
    .then((data) => {
      log.info(data.filePaths);
      if (data.filePaths.length > 0) {
        event.sender.send("selected-directory", data.filePaths[0]);
      }
    });
});

ipcMain.on("open-singlefile-dialog", (event) => {
  dialog
    .showOpenDialog({
      properties: ["openFile", "multiSelections"],
      filters: [{ name: "Images", extensions: ["tif", "tiff"] }],
    })
    .then((data) => {
      if (data.filePaths.length > 0) {
        event.sender.send("selected-file", data.filePaths);
      }
    });
});

const desktop_fingerprint = crypto.randomBytes(16).toString("hex");

ipcMain.on("getFingerprint", (event) => {
  event.sender.send("getFingerprint", desktop_fingerprint);
});

ipcMain.on("messagebox", () => {
  dialog.showMessageBox({
    type: "info",
    title: "Info",
    message:
      "Search queries using more than one image as reference is not yet implemented. ",
    detail: "Put an issue on GitHub if this bothers you.",
  });
});

ipcMain.on("waitUntilCalcFeaturesFinishedbox", () => {
  dialog.showMessageBox({
    type: "info",
    title: "Indexing in progress..",
    message: "You can search the library once the indexing has finished. ",
    detail: "In the meantime you can add or remove folders.",
  });
});

ipcMain.on("emptyResultBox", () => {
  dialog.showMessageBox({
    type: "info",
    title: "No results",
    message: "The engine returned no results for the selected images.",
    detail:
      "ML-SIM expects TIFF stacks with at least 9 frames. The engine log has the details. If you did not expect this, file a bug on the ML-SIM GitHub issue tracker.",
  });
});

function createWindow() {
  mainWindow = createAppWindow();
  mainWindow.on("closed", () => (mainWindow = null));
  global.mainWindow = mainWindow;
}

app.on("ready", createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("activate", () => {
  if (mainWindow === null) {
    createWindow();
  }
});
