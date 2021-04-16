const stream = require("stream");
const { promisify } = require("util");
const fs = require("fs");
const path = require("path");
const got = require("got");
const crypto = require("crypto");

const pipeline = promisify(stream.pipeline);

let baseurl = "https://ml-sim.s3.eu-west-2.amazonaws.com/pdist";
let total_downloaded = 0;
let total_to_download = 0;
let prev_val = null;
let prev_time = null;
let mbps = "N/A";
let downloads_in_progress = 0;
let setup_models = [];
var modeldir = path.join("pyengine", "models");

let pdist_json = null;

const get_latest_json = async () => {
  const response = await got(baseurl + "/latest.json");
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

function download(url, dest, post_func) {
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
        console.log("calling post function");
        post_func();
      }
    } catch (error) {
      console.log("error:", error);
    }
  })();
}

function print_progress() {
  if (downloads_in_progress > 0) {
    if (!prev_time) {
      prev_time = new Date().getMilliseconds();
      prev_val = total_downloaded;
    } else {
      mbps = (total_downloaded - prev_val) / 1000000;
      mbps = "" + mbps.toFixed(1) + " MB/s";
      prev_val = total_downloaded;
      prev_time = new Date().getMilliseconds();
    }
    let percent = (100 * total_downloaded) / total_to_download;
    process.stdout.write(
      "Downloading:\t" + percent.toFixed(1) + " %,\t" + mbps + "\r"
    );
    setTimeout(() => {
      print_progress();
    }, 1000);
  } else {
    console.log("Nothing more to download");
  }
}

function download_mdls() {
  //* model download loop
  setup_models.forEach((model) => {
    let modelpath = path.join(modeldir, model.url);
    download(baseurl + "/models/" + model.url, modelpath, () => {
      console.log("Downloading", model.url);
      checkHash(modelpath, (hashval) => {
        if (model.hash === hashval) {
          //* downloaded model has valid hash
          console.log("Downloaded", model.url, "hash valid");

          if (downloads_in_progress === 0) {
            //* downloading of models has finished
            console.log("Downloading completed");
          }
        } else {
          //* downloaded model has invalid hash
          console.log("Downloaded", model.url, "invalid - relaunch?");
        }
      });
    });
  });

  if (setup_models.length > 0) {
    setTimeout(() => {
      print_progress();
    }, 1000);
  }
}

function check_and_download() {
  (async () => {
    try {
      console.log("Getting json", baseurl + "/latest.json");
      if (pdist_json == null) pdist_json = await get_latest_json();
      console.log(pdist_json)
      let mdls = pdist_json.models;
      let chk_mdls = 0;
      if (!fs.existsSync(modeldir)) fs.mkdirSync(modeldir);

      //* model integrity
      mdls.forEach((model) => {
        let modelpath = path.join(modeldir, model.url);
        if (!fs.existsSync(modelpath)) {
          console.log("model", model, "does not exist - will download");
          setup_models.push(model);
          if (++chk_mdls === mdls.length) download_mdls();
        } else {
          // check model hash
          checkHash(modelpath, (hashval) => {
            if (model.hash !== hashval) {
              console.log(model, "HASH MISMATCH - will download");
              setup_models.push(model);
            } else {
              console.log(model.url, "HASH MATCH");
            }
            if (++chk_mdls === mdls.length) download_mdls();
          });
        }
      });

    } catch (error) {
      console.log("error:", error);
    }
  })();
}

check_and_download();
