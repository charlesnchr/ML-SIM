const electron = require("electron");
const log = require("electron-log");

const app = electron.app;
const appData = app.getPath("appData");
const fs = require("fs");
const rimraf = require("rimraf");
const path = require("path");
const crypto = require("crypto");
const isDev = require("electron-is-dev");

const process = require("process");
const got = require("got");
const unzipper = require("unzipper");
const { ipcMain, dialog } = require("electron");

const stream = require("stream");
const { promisify } = require("util");
const pipeline = promisify(stream.pipeline);

const Store = require("electron-store");
const store = new Store();

const engineCommunication = require("./EngineCommunication.js");

let baseurl = "https://ml-sim.s3.eu-west-2.amazonaws.com/pdist";
if (process.env.pdist_server) {
  baseurl = process.env.pdist_server;
}


/*************************************************************
 * start Python socket server
 *************************************************************/

//* Parameters used in the following
var use_pysrc = isDev && !process.env.use_pdist;
let setup_pdist = false;
let setup_models = [];
let total_downloaded = 0;
let total_to_download = 0;
let downloads_in_progress = 0;
let extraction_in_progress = false;
let pythonProgramPath;
let enginedir = path.join(appData, "ML-SIM-Engine");
let enginebin = "engine.exe"; //! Windows specific
var modeldir = path.join(enginedir, "models");
let archivefullpath = path.join(enginedir, "engine.mlsim");
let skip_integrity_check = store.get("skip_integrity_check");
let pdist_json = null;

  

//*
//* Other functions
//*

const get_latest_json = async () => {
    const response = await got(baseurl + "/latest.json");
    log.info("latest.json", response.body);
    return JSON.parse(response.body);
  };
  
  var checkHash = (filepath, callback) => {
    var fd = fs.createReadStream(filepath);
    var hash = crypto.createHash("sha1");
    hash.setEncoding("hex");
    fd.on("end", function() {
      hash.end();
      let hashval = hash.read();
      callback(hashval);
    });
  
    fd.pipe(hash);
  };
  
  var extract = (hashval) => {
    extraction_in_progress = true;
  
    fs.createReadStream(archivefullpath)
      .pipe(
        unzipper.Extract({
          path: path.join(enginedir, hashval),
        })
      )
      .on("close", () => {
        log.info("extraction complete");
        extraction_in_progress = false;
        store.set("engine_hash", hashval);
        if (downloads_in_progress == 0) {
          log.info("Engine setup complete");
          store.set("pdist_version", pdist_json.version);
          engineCommunication.createPyProc(use_pysrc, pythonProgramPath);
          return;
        }
      });
  };
  
  var showHashMismatchDialog = () => {
    const options = {
      type: "warning",
      buttons: ["Relaunch now", "Dismiss"],
      defaultId: 0,
      title: "An error occurred during installation of the engine",
      message:
        "The files that were downloaded do not match the signatures of the ones on our server. Please try again by relaunching.",
      detail:
        "File a bug on the ML-SIM GitHub Issues tracker if this keeps happening.",
    };
  
    dialog.showMessageBox(null, options, (response) => {
      if (response == 0) {
        app.relaunch();
        app.exit(0);
      }
    });
  };
  
  function print_progress() {
    if (downloads_in_progress > 0) {
      log.info("Downloading status", total_downloaded / total_to_download);
      setTimeout(() => {
        print_progress();
      }, 20000);
    } else {
      log.info("nothing to download");
    }
  }
  
  function download(url, dest, post_func) {
    log.info("download", url, dest);
    let t0 = new Date().getTime();
    let added_to_total = false;
    let last_downloaded = 0;
    downloads_in_progress++;
  
    (async () => {
      try {
        await pipeline(
          got.stream(url).on("downloadProgress", (progress) => {
            if (!added_to_total) {
              added_to_total = true;
              total_to_download += progress.total;
              total_downloaded += progress.transferred;
              last_downloaded = progress.transferred;
            } else {
              total_downloaded += progress.transferred - last_downloaded;
              last_downloaded = progress.transferred;
            }
          }),
          fs.createWriteStream(dest)
        );
        downloads_in_progress--;
        if (post_func) {
          log.info("calling post function");
          post_func();
        }
      } catch (error) {
        log.info("error:", error);
      }
    })();
  }
  
  var download_checkHash_extract = (url, dest) => {
    download(url, dest, () => {
      log.info("downloaded latest archive");
      checkHash(dest, (hashval) => {
        if (hashval === engine_hash) {
          log.info("hash of new archive is valid");
          extract(hashval); // will overwrite if exists
        } else {
          log.warn("hash of new archive is invalid - ask user to relaunch");
          showHashMismatchDialog();
        }
      });
    });
  };
  
  const config_push_model = (modelname) => {
    let engine_models = store.get("engine_models");
    if (engine_models == null || engine_models.length === 0) {
      engine_models = [modelname];
    } else {
      let model_exists = false;
      engine_models.forEach((engine_model) => {
        if (engine_model === modelname) model_exists = true;
      });
      if (!model_exists) engine_models.push(modelname);
      else log.info("model", modelname, "already in config");
    }
    store.set("engine_models", engine_models);
  };
  
  const setupEngine = async () => {
    //* Download, extract, and launch
  
    log.info("baseurl", baseurl);
  
    if (!fs.existsSync(enginedir)) fs.mkdirSync(enginedir);
  
    (async () => {
      try {
        log.info("Attempting to download json", baseurl + "/latest.json");
        if (pdist_json == null) pdist_json = await get_latest_json();
        engine_hash = pdist_json.engine.hash;
        pythonProgramPath = path.join(enginedir, engine_hash, enginebin);
  
        if (setup_pdist) {
          //*  clean up
          log.info("Starting async clean up task");
          (async () => {
            try {
              fs.readdirSync(enginedir).forEach((pathname) => {
                if (pathname !== "engine.mlsim" && pathname !== "models") {
                  const stat = fs.statSync(path.join(enginedir, pathname));
                  if (stat.isFile()) {
                    log.info("Removing file", pathname);
                    fs.unlink(path.join(enginedir, pathname), (err) => {
                      if (err) throw err;
                      console.log("Deleted", pathname);
                    });
                  } else {
                    log.info("Removing folder", path.join(enginedir, pathname));
                    rimraf(path.join(enginedir, pathname), function() {
                      log.info("Deleted", pathname);
                    });
                  }
                }
              });
            } catch (err) {
              log.info("Could not read", enginedir, err);
            }
          })();
  
          //* download pdist
          log.info("downloading pdist");
          if (fs.existsSync(archivefullpath)) {
            checkHash(archivefullpath, (hashval) => {
              if (hashval === engine_hash) {
                log.info(
                  "Latest archive found present",
                  hashval,
                  "- starting extraction"
                );
                extract(pdist_json.engine.hash);
              } else {
                log.info(
                  "Hash of present archive does not match, new vs old",
                  engine_hash,
                  hashval,
                  "- downloading latest"
                );
                fs.unlinkSync(archivefullpath);
                download_checkHash_extract(
                  baseurl + "/" + pdist_json.engine.url,
                  archivefullpath
                );
              }
            });
          } else {
            log.info("archive does not exist, dowloading afresh");
            download_checkHash_extract(
              baseurl + "/" + pdist_json.engine.url,
              archivefullpath
            );
          }
        }
  
        //* check if models are present
        if (!fs.existsSync(modeldir)) fs.mkdirSync(modeldir);
        log.info("looping over models", pdist_json.models);
  
        //* model download loop
        setup_models.forEach((model) => {
          let modelpath = path.join(modeldir, model.url);
          download(baseurl + "/models/" + model.url, modelpath, () => {
            checkHash(modelpath, (hashval) => {
              if (model.hash === hashval) {
                //* downloaded model has valid hash
                log.info("Downloaded", model.url, "hash valid");
                config_push_model(model.url);
  
                //* should subprocess be started
                if (downloads_in_progress === 0) {
                  // downloading of models has finished
                  if (!extraction_in_progress) {
                    log.info("Engine setup complete");
                    store.set("pdist_version", pdist_json.version);
                    engineCommunication.createPyProc(use_pysrc, pythonProgramPath);
                    return;
                  }
                }
              } else {
                //* downloaded model has invalid hash
                log.warn("Downloaded", model.url, "invalid - relaunch?");
                showHashMismatchDialog();
              }
            });
          });
        });
  
        //* start progress viewing
        setTimeout(() => {
          print_progress();
        }, 1000);
      } catch (error) {
        log.info("error:", error);
      }
    })();
  };
  
  const initEngine = async () => {
    // attach listener (app ready event inside initEngine)
    ipcMain.on("EngineStatus", (event, json) => {
      if (engineCommunication.get_serverActive()) {
        event.sender.send("EngineStatus", "a");
      } else if (downloads_in_progress > 0) {
        let frac = parseFloat(total_downloaded) / parseFloat(total_to_download);
        event.sender.send("EngineStatus", "d," + parseInt(100 * frac));
      } else if (extraction_in_progress) {
        event.sender.send("EngineStatus", "e");
      } else {
        event.sender.send("EngineStatus", "w"); // something else, just wait
      }
    });
  
    //* run src or compiled
    if (use_pysrc) {
      pythonProgramPath = path.join(
        app.getAppPath(),
        "pyengine",
        "pysrc",
        "engine.py"
      );
      engineCommunication.createPyProc(use_pysrc, pythonProgramPath);
      return;
    }
  
    const runOrSetupEngine = () => {
      if (setup_pdist === false && setup_models.length === 0) {
        log.info("nodownloadsneeded-startingnow");
  
        engineCommunication.createPyProc(use_pysrc, pythonProgramPath);
        return;
      } else {
        setupEngine();
      }
    };
  
    //* downloading latest json
    pdist_json = await get_latest_json();
    let str_v = store.get("pdist_version");
  
    if (!skip_integrity_check || pdist_json.version !== str_v) {
      //* integrity check - will be skipped for faster launching
      log.info("Performing integrity check");
      engine_hash = pdist_json.engine.hash;
      pythonProgramPath = path.join(enginedir, engine_hash, enginebin);
      let chk_mdls = 0;
      let chk_engine = false;
      let mdls = pdist_json.models;
      setup_pdist = true; // force re-extraction (for robustness)
  
      //* engine
      // check existence
      if (!fs.existsSync(archivefullpath)) {
        log.info("engine archive does not exist - will download");
        chk_engine = true;
        if (chk_mdls === mdls.length) runOrSetupEngine();
      } else {
        // check engine hash
        checkHash(archivefullpath, (hashval) => {
          if (pdist_json.engine.hash !== hashval) {
            log.info(
              "new vs old archive hash mismatch - will download",
              pdist_json.engine.hash,
              hashval
            );
            log.info("deleting old archive");
            fs.unlinkSync(archivefullpath);
            store.set("engine_hash", null);
          } else {
            log.info("archive hash match, but will reinstall", hashval);
            store.set("engine_hash", null);
          }
          chk_engine = true;
          if (chk_mdls === mdls.length) runOrSetupEngine();
        });
      }
  
      //* model integrity
      mdls.forEach((model) => {
        let modelpath = path.join(modeldir, model.url);
        if (!fs.existsSync(modelpath)) {
          log.info("model", model, "does not exist - will download");
          setup_models.push(model);
          if (++chk_mdls === mdls.length && chk_engine) runOrSetupEngine();
        } else {
          // check model hash
          checkHash(modelpath, (hashval) => {
            if (model.hash !== hashval) {
              log.warn(model, "hash mismatch - will download");
              setup_models.push(model);
            } else {
              log.info(model.url, "hash match");
              config_push_model(model.url);
            }
            if (++chk_mdls === mdls.length && chk_engine) runOrSetupEngine();
          });
        }
      });
    } else {
      //* quick check - look at stored hash vs online
      //* should pdist be downloaded
      let engine_hash = store.get("engine_hash");
      if (!engine_hash) {
        setup_pdist = true;
        log.info("nohash,willdownload");
      } else if (engine_hash !== pdist_json.engine.hash) {
        setup_pdist = true;
        log.info("stored hash mismatch,willdownload", engine_hash);
      } else {
        log.info("Stored engine_hash", engine_hash);
        pythonProgramPath = path.join(enginedir, engine_hash, enginebin);
        if (!fs.existsSync(pythonProgramPath)) {
          log.info("engine folder does not exist");
          setup_pdist = true;
        }
      }
  
      //* should models be downloaded
      let pdist_models = pdist_json.models;
      let engine_models = store.get("engine_models");
      if (!engine_models) {
        setup_models = pdist_models;
        log.info("downloadallmodels", setup_models);
      } else {
        pdist_models.forEach((model) => {
          let modelpath = path.join(enginedir, "models", model.url);
          if (!fs.existsSync(modelpath)) {
            setup_models.push(model);
          }
        });
      }
  
      runOrSetupEngine();
    }
  };


  module.exports = {
      initEngine
  }
