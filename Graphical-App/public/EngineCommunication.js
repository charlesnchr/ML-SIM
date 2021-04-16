const electron = require("electron");
const app = electron.app;
const path = require("path");
const log = require("electron-log");
var net = require("net");
const { ipcMain, dialog } = require("electron");
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
  client.connect(5002, "127.0.0.1", function() {
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
      require("child_process").exec('explorer.exe "' + newdir + '"');
      client.write("0"); // send more
    } else if (data[0] == "z") {
      log.info("Received: ", data);
      let newfile = data.substr(1, data.length);
      require("child_process").exec('explorer.exe /select,"' + newfile + '"');
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
    log.info("Will perform integrity check program start");
    store.set("skip_integrity_check", null);

    const options = {
      type: "warning",
      buttons: ["Relaunch now", "Dismiss"],
      defaultId: 0,
      title: "An error occurred",
      message: "Please relaunch the program to try to recover. ",
      detail:
        "File a bug on the ML-SIM GitHub Issues tracker if this keeps happening.",
    };

    dialog.showMessageBox(null, options, (response) => {
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
let pyProc = null;
const createPyProc = (use_pysrc, script) => {
  let port = "5002";
  let useCloud = store.get("useCloud") ? "1" : "0";
  let cachedir = store.get("cachedir");

  if (!use_pysrc) {
    log.info("using engine", script);
    pyProc = require("child_process").execFile(
      script,
      [port, useCloud, cachedir],
      { cwd: path.dirname(script) }
    );
  } else {
    log.info(
      "spawning python server",
      +"pipenv run python",
      [script, port, useCloud, cachedir],
      { cwd: path.dirname(script) }
    );
    pyProc = require("child_process").spawn(
      "pipenv",
      ["run", "python", script, port, useCloud, cachedir],
      { cwd: path.dirname(script) }
    );
  }

  if (pyProc != null) {
    //log.info(pyProc)
    log.info("child process success on port " + port);
    log.info("now waiting for Python server");
    waitForPython();
  }
};

const exitEngine = () => {
  log.info("exited Python socket server");
  pyProc.kill();
  pyProc = null;
};

const waitForPython = () => {
  const client = new net.Socket();
  const tryConnection = () =>
    client.connect({ port: 5002 }, () => {
      client.write("nothing");
      client.end();
      log.info("server is now active");
      if (!store.get("skip_integrity_check"))
        store.set("skip_integrity_check", true);
      serverActive = true;
    });
  tryConnection();
  client.on("error", (error) => {
    log.info("retrying connection");
    if (!serverActive) setTimeout(tryConnection, 100);
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
