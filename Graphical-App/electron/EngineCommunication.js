const { app, ipcMain, dialog, shell } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const log = require("electron-log/main");
var net = require("net");
const Store = require("electron-store");
const store = new Store();

/*************************************************************
 * Nodejs socket client
 *************************************************************/
var serverActive = false;
var filepaths = [];
var path_send_count = 0; // elements transmitted

const sendToPython = (event, json) => {
  if (!serverActive) {
    log.warn("Engine not ready -", json["cmd"], "ignored");
    return;
  }
  // if (json["cmd"] == "calcFeatures" && path_send_count > 0) return; // finishing transmitting current batch

  var client = new net.Socket();
  client.connect(port, "127.0.0.1", function() {
    log.info("here",json);
    if ("filepaths" in json) {
      // start chunked sending of filepaths
      filepaths = json["filepaths"];
      var msg = json["cmd"] + "\n" + json["arg"];
      "\n" + filepaths.length;
      log.info("To socket", msg);
      client.write(msg);
    } else {
      var msg = json["cmd"] + "\n" + json["arg"];
      log.info("To socket", msg);
      client.write(msg);
    }
  });

  client.on("data", function(data) {
    data = data.toString("utf-8");
    if (data[0] == "2") {
      // received result paths
      log.info("Received results: ", data.substr(0, 20), "...");
      if (event) {
        log.info("now sending back");
        event.sender.send("ReceivedResults", data.substr(1, data.length));
      }
      client.write("0"); // send more
    } else if (data[0] == "s") {
      log.info("Received: ", data);
      // processing status
      event.sender.send("status", data[1], data.substr(2, data.length));
      client.write("0"); // send more
    } else if (data[0] == "t") {
      log.info("Received thumb: ", data);
      // processing status
      let res = data.split("\n");
      let filepath = res[1];
      let thumbpath = res[2];
      let dim = res[3];
      event.sender.send("thumb_" + filepath, thumbpath, dim);
      client.write("0"); // send more
    } else if (data[0] == "e") {
      log.info("Received: ", data);
      let newdir = data.substr(1, data.length);
      shell.openPath(newdir);
      client.write("0"); // send more
    } else if (data[0] == "z") {
      log.info("Received: ", data);
      let newfile = data.substr(1, data.length);
      shell.showItemInFolder(newfile);
      client.write("0"); // send more
    } else if (data[0] == "p") {
      // request to send filepaths
      if (path_send_count < filepaths.length) {
        var res_chunk = filepaths.slice(path_send_count, path_send_count + 7);
        path_send_count += 7;
        client.write(res_chunk.join("\n"));
      } else {
        log.info("finished sending all file paths");
        client.write("x"); // tell server there are no more paths to send
        path_send_count = 0;
      }
    } else if (data[0] == "1") {
      // kill signal, communication completed
      client.write("1");
      client.destroy(); // kill client after server's response
    }
  });

  client.on("close", function() {
    if (json["cmd"] == "calcFeatures") {
      path_send_count = 0;
    }
    log.info("Connection closed");
  });

  client.on("error", (error) => {
    log.warn("Error in socket communication", error);
    dialog
      .showMessageBox({
        type: "warning",
        buttons: ["Relaunch now", "Dismiss"],
        defaultId: 0,
        title: "An error occurred",
        message: "Please relaunch the program to try to recover. ",
        detail:
          "File a bug on the ML-SIM GitHub Issues tracker if this keeps happening.",
      })
      .then(({ response }) => {
        if (response == 0) {
          app.relaunch();
          app.exit(0);
        }
      });
  });
};

//*
//* Subprocess function
//*
let port = null; // chosen when the engine starts
let pyProc = null;
let quitting = false;

// Ask the operating system for a free local port.
const getFreePort = () =>
  new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const freePort = server.address().port;
      server.close(() => resolve(freePort));
    });
  });

const createPyProc = async ({ python, script, modelDir, onFailure }) => {
  port = await getFreePort();
  const useCloud = store.get("useCloud") ? "1" : "0";
  const cachedir = store.get("cachedir");

  log.info("starting engine:", python, script);
  pyProc = spawn(python, [script, String(port), useCloud, cachedir], {
    cwd: path.dirname(script),
    env: {
      ...process.env,
      MLSIM_MODELS_DIR: modelDir,
      PYTHONUNBUFFERED: "1",
    },
  });

  let lastOutput = "";
  const forward = (data) => {
    const text = data.toString().trimEnd();
    if (text) {
      lastOutput = text;
      log.info("[engine]", text);
    }
  };
  pyProc.stdout.on("data", forward);
  pyProc.stderr.on("data", forward);

  const setupHint =
    ' Run "npm run setup-engine" in Graphical-App, or set MLSIM_PYTHON to a Python that has the packages in pyengine/requirements.txt.';

  pyProc.on("error", (err) => {
    log.error("could not start the engine", err);
    pyProc = null;
    onFailure("Could not run " + python + "." + setupHint);
  });

  pyProc.on("exit", (code) => {
    serverActive = false;
    pyProc = null;
    if (quitting) return;
    log.error("engine exited with code", code);
    const reason = lastOutput.split("\n").pop();
    onFailure(
      "The Python engine stopped" + (reason ? " (" + reason + ")." : ".") + setupHint
    );
  });

  log.info("now waiting for Python server on port " + port);
  waitForPython();
};

const exitEngine = () => {
  quitting = true;
  if (pyProc != null) {
    pyProc.kill();
    pyProc = null;
    log.info("exited Python socket server");
  }
};

const waitForPython = () => {
  if (serverActive || pyProc == null) return;
  const client = new net.Socket();
  client.on("error", () => {
    client.destroy();
    setTimeout(waitForPython, 200);
  });
  client.connect({ port: port, host: "127.0.0.1" }, () => {
    client.write("nothing");
    client.end();
    log.info("server is now active");
    serverActive = true;
  });
};

ipcMain.on("sendToPython", (event, json) => {
  log.info(json["cmd"]);
  sendToPython(event, json);
});

ipcMain.on("isServerActive", (event) => {
  if (serverActive) event.sender.send("serverActive");
});

const get_serverActive = () => {
  return serverActive
}

module.exports = {
  createPyProc,
  exitEngine,
  get_serverActive,
};
