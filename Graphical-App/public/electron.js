const electron = require("electron");
const app = electron.app;
// const Tray = electron.Tray;
// const Menu = electron.Menu;
const BrowserWindow = electron.BrowserWindow;

const path = require("path");
const isDev = require("electron-is-dev");

const process = require("process");
const url = require("url");

let mainWindow;
const { ipcMain, dialog } = require("electron");
const crypto = require("crypto");

const { autoUpdater } = require("electron-updater");
var quitAndInstallReady = false;
var updateDetected = false;
const Store = require("electron-store");
const store = new Store();

const log = require("electron-log");

const appData = app.getPath("appData");

let exportdir = store.get("exportdir");
if (exportdir == null) {
  exportdir = path.join(app.getPath("documents"), "Mambio");
  store.set("exportdir", exportdir);
}
let cachedir = store.get("cachedir");
if (cachedir == null) {
  cachedir = path.join(appData, "Mambio-Library", "2");
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

// file dialog handler
// ipcMain.on('renderMenu', (event, profilename, settingsHandler) => {
//   rendermenu(profilename,settingsHandler,event)
// })



ipcMain.on("showImage", (event, imgsrc) => {
  let webPreferences = {
    devTools: false,
    nodeIntegration: true,
    webSecurity: false, // is there another way to get images to load?
  };
  if (isDev) webPreferences.devTools = true;

  let imageWindow = new BrowserWindow({
    width: 1000,
    height: 700,
    minWidth: 700,
    minHeight: 500,
    title: "ML-SIM",
    webPreferences: webPreferences,
    titleBarStyle: "hidden",
  });
  imageWindow.setBackgroundColor("#444444");
  imageWindow.setMenuBarVisibility(false);

  let imageViewURL = url.format({
    pathname: path.join(__dirname, "../build/index.html"),
    hash: "imageview",
    protocol: "file",
    slashes: false,
  });

  if(isDev && !process.env.RunReactCompile) {

  } else {
    log.info('will go to',imageViewURL);
  }
  imageWindow.loadURL(
    isDev && !process.env.RunReactCompile
      ? "http://localhost:4001#/imageview"
      : imageViewURL
  );


  imageWindow.webContents.on("dom-ready", (event) => {
    log.info('before noramlise',imgsrc);
    log.info('after normalize',path.normalize(imgsrc));
    imageWindow.webContents.send("filepathArgument", imgsrc.replace(/\\/g, "/"));
  });
});

ipcMain.on("open-file-dialog", (event) => {
  dialog
    .showOpenDialog({
      properties: ["openFile", "openDirectory"],
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

let desktop_fingerprint = "-1";
crypto.randomBytes(16, function(err, buffer) {
  desktop_fingerprint = buffer.toString("hex");
});

ipcMain.on("getFingerprint", (event) => {
  event.sender.send("getFingerprint", desktop_fingerprint);
});


//  message box
ipcMain.on("messagebox", (event) => {
  const options = {
    type: "info",
    buttons: [],
    defaultId: 0,
    title: "Info",
    message:
      "Search queries using more than one image as reference is not yet implemented. ",
    detail: "Put an issue on GitHub if this bothers you.",
  };

  dialog.showMessageBox(null, options, (response) => {
    log.info(response);
  });
});

//  message box
ipcMain.on("waitUntilCalcFeaturesFinishedbox", (event) => {
  const options = {
    type: "info",
    buttons: [],
    defaultId: 0,
    title: "Indexing in progress..",
    message: "You can search the library once the indexing has finished. ",
    detail:
      "In the meantime you can add or remove folders. If the indexing is taking a long time, try out the cloud compute.",
  };

  dialog.showMessageBox(null, options, (response) => {
    log.info(response);
  });
});

//  message box
ipcMain.on("emptyResultBox", (event) => {
  const options = {
    type: "info",
    buttons: [],
    defaultId: 0,
    title: "Missing results for search query",
    message: "No valid results were found for your search query.",
    detail:
      "If you did not expect this, file a bug on the ML-SIM Github Issues tracker.",
  };

  dialog.showMessageBox(null, options, (response) => {
    log.info(response);
  });
});

function createWindow() {
  let webPreferences = {
    devTools: false,
    nodeIntegration: true,
    webSecurity: false, // is there another way to get images to load?
  };
  if (isDev) webPreferences.devTools = true;

  mainWindow = new BrowserWindow({
    width: 1000,
    height: 700,
    minWidth: 700,
    minHeight: 500,
    title: "ML-SIM",
    webPreferences: webPreferences,
    titleBarStyle: "hidden",
  });

  mainWindow.loadURL(
    isDev && !process.env.RunReactCompile
      ? "http://localhost:4001"
      : `file://${path.join(__dirname, "../build/index.html")}`
  );

  mainWindow.on("closed", () => (mainWindow = null));
  mainWindow.setBackgroundColor("#444444");

  global.mainWindow = mainWindow;
  // BrowserWindow.addDevToolsExtension(path.join('C:/Users/charl/AppData/Local/Google/Chrome/User Data/Default/Extensions/fmkadmapgofadopljbjfkapdkoienihi/4.4.0_0'));
  // BrowserWindow.removeDevToolsExtension(name)
  // name given by: BrowserWindow.getDevToolsExtensions
}

app.on("ready", createWindow);

app.on("window-all-closed", () => {
  if (quitAndInstallReady) {
    autoUpdater.quitAndInstall(false, true); // isSilent, isForceRunAfter (ignored when isSilent is false)
  } else {
    if (!engineSetup.serverActive) {
      store.set("skip_integrity_check", false); // there may be a problem
    }

    if (process.platform !== "darwin") {
      app.quit();
    }
  }
});

app.on("activate", () => {
  if (mainWindow === null) {
    createWindow();
  }
});

/*************************************************************
 * App update
 *************************************************************/

ipcMain.on("startUpdateService", (event) => {
  autoUpdater.on("checking-for-update", () => {});

  autoUpdater.on("update-available", (info) => {
    log.info("Update available", info);
    event.sender.send("update-available", info);
    updateDetected = true;
  });
  autoUpdater.on("update-not-available", (info) => {
    log.info("Update not available.");
  });
  autoUpdater.on("error", (err) => {
    log.info("Error in auto-updater. " + err);
  });
  autoUpdater.on("download-progress", (progressObj) => {
    event.sender.send("download-progress", progressObj);
    let log_message = "Download speed: " + progressObj.bytesPerSecond;
    log_message = log_message + " - Downloaded " + progressObj.percent + "%";
    log_message =
      log_message +
      " (" +
      progressObj.transferred +
      "/" +
      progressObj.total +
      ")";
    log.info(log_message);
  });

  autoUpdater.on("update-downloaded", (info) => {
    log.info("Update downloaded");
    event.sender.send("update-downloaded", info);
    quitAndInstallReady = true;
    store.set("skip_integrity_check", false);

    const options = {
      type: "info",
      buttons: ["Relaunch now", "Later"],
      defaultId: 0,
      title: "ML-SIM update ready to install",
      message: "Do you want to begin installation of the update?",
      detail:
        "File a bug on the ML-SIM GitHub Issues tracker if you have any trouble updating.",
    };

    dialog.showMessageBox(null, options, (response) => {
      if (response == 0) {
        autoUpdater.quitAndInstall(false, true); // isSilent, isForceRunAfter (ignored when isSilent is false)
      }
    });
  });

  function checkUpdate() {
    if (!updateDetected) {
      autoUpdater.checkForUpdates();
      setTimeout(function() {
        checkUpdate();
      }, 10000);
    }
  }

  autoUpdater.logger = log;
  autoUpdater.logger.transports.file.level = "info";
  log.info("Starting update service");
  checkUpdate();
});
