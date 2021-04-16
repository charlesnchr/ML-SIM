var fs = window.require("fs");
var path = window.require("path");
var crypto = window.require("crypto");
const log = window.require("electron-log");

var sess = require("./sess.js");

function checkFilters(fn) {
  let returnval = true;

  if (sess.excludePattern !== "") {
    var pats = sess.excludePattern.split(",");
    for (let index = 0; index < pats.length; index++) {
      const pat = pats[index];
      if (pat === "") continue;
      if (fn.includes(pat)) {
        return false;
      }
    }
  }

  if (sess.includePattern !== "") {
    returnval = false;
    var pats = sess.includePattern.split(",");
    for (let index = 0; index < pats.length; index++) {
      const pat = pats[index];
      if (pat === "") continue;
      if (fn.includes(pat)) {
        returnval = true;
      }
    }
  }

  return returnval;
}

function readFilesRecursively(input_dirs, exts, sortBy, callBack) {
  // Recursive function
  var walk = function(root, dir, done) {
    var results = [];

    fs.readdir(dir, function(err, list) {
      if (err) return done(err);
      var pending = list.length;
      if (!pending) return done(null, results);
      list.forEach(function(file) {
        const name = path.parse(file).name;
        const ext = path.parse(file).ext;
        const filepath = path.resolve(dir, file);
        const stat = fs.statSync(filepath);
        if (stat.isDirectory()) {
          walk(root, filepath, function(err, res) {
            results = results.concat(res);
            if (!--pending) done(null, results);
          });
        } else {
          if (stat.isFile() && exts.indexOf(ext) >= 0)
            if (checkFilters(filepath))
              results.push({ filepath, name, ext, stat, root });
          if (!--pending) done(null, results);
        }
      });
    });
  };

  // Outer loop
  var pending = input_dirs.length;
  var files_dict = {};
  var nfiles = Array(input_dirs.length);
  let dirs = input_dirs.slice(); // copy by value
  let basedirs = input_dirs.map((dir) => path.basename(dir));

  dirs.forEach(function(dir) {
    walk(dir, dir, function(err, res) {
      files_dict[dir] = res;
      nfiles[dirs.indexOf(dir)] = res.length;
      if (!--pending) {
        var files = [];
        dirs.forEach(function(dir) {
          files = files.concat(files_dict[dir]);
        });
        if (files.length === 0) callBack({ filepaths: [], dirsizes: [] });

        if (sortBy && sortBy === "folderOrder") {
          log.info("sorting by folderorder");
          files.sort((a, b) => {
            let aidx = basedirs.indexOf(path.basename(a.root));
            let bidx = basedirs.indexOf(path.basename(b.root));
            if (aidx == bidx) {
              return a.filepath.localeCompare(b.filepath, undefined, {
                numeric: true,
                sensitivity: "base",
              });
            } else {
              return aidx > bidx ? 1 : -1;
            }
          });
          log.info("sorted now", files[0]);
        } else if (sortBy && sortBy === "reverseFolderOrder") {
          log.info("sorting by reverse folderorder");
          files.sort((a, b) => {
            let aidx = basedirs.indexOf(path.basename(a.root));
            let bidx = basedirs.indexOf(path.basename(b.root));
            if (aidx == bidx) {
              return a.filepath.localeCompare(b.filepath, undefined, {
                numeric: true,
                sensitivity: "base",
              });
            } else {
              return bidx > aidx ? 1 : -1;
            }
          });
          log.info("sorted now", files[0]);
        } else if (sortBy && sortBy === "dateCreated") {
          log.info("sorting by datecreated folderorder");
          files.sort((a, b) => {
            return a.birthtimeMs > b.birthtimeMs;
          });
        } else {
          files.sort((a, b) => {
            // natural sort alphanumeric strings
            // https://stackoverflow.com/a/38641281
            return a.filepath.localeCompare(b.filepath, undefined, {
              numeric: true,
              sensitivity: "base",
            });
          });
        }

        callBack({
          filepaths: files.map(function(file) {
            return file.filepath;
          }),
          dirsizes: nfiles,
        });
      }
    });
  });
}

function readFilesSync(input_dirs, exts, sortBy) {
  const files = [];
  const nfiles = [];
  let dirs = input_dirs.slice(); // copy by value

  if (sortBy && sortBy === "reverseFolderOrder") {
    dirs.reverse();
  }

  dirs.forEach((dir) => {
    var count = 0;
    try {
      fs.readdirSync(dir).forEach((filename) => {
        try {
          const name = path.parse(filename).name;
          const ext = path.parse(filename).ext;
          const filepath = path.resolve(dir, filename);
          const stat = fs.statSync(filepath);
          const isFile = stat.isFile();
          if (isFile && exts.indexOf(ext) >= 0)
            if (checkFilters(filepath))
              files.push({ filepath, name, ext, stat });
          count++;
        } catch (err) {
          log.info("Error reading", filename, "  -  Error message:", err);
        }
      });
      nfiles.push(count);
    } catch (err) {
      log.info("Could not read", dir);
    }
  });

  if (files.length === 0) return { filepaths: [], dirsizes: [] };

  if (sortBy && sortBy === "folderOrder") {
  } else if (sortBy && sortBy === "reverseFolderOrder") {
  } else if (sortBy && sortBy === "dateCreated") {
    files.sort((a, b) => {
      return a.stat.birthtimeMs > b.stat.birthtimeMs;
    });
  } else {
    files.sort((a, b) => {
      // natural sort alphanumeric strings
      // https://stackoverflow.com/a/38641281
      return a.name.localeCompare(b.name, undefined, {
        numeric: true,
        sensitivity: "base",
      });
    });
  }

  return {
    filepaths: files.map(function(file) {
      return file.filepath;
    }),
    dirsizes: nfiles,
  };
}

function getHashOfArray(inputArray) {
  var tmp_arr = inputArray.slice(0, inputArray.length);
  tmp_arr.sort();
  var inputArray_str = tmp_arr.join("");
  var filepaths_hash = crypto
    .createHash("md5")
    .update(inputArray_str)
    .digest("hex");
  return filepaths_hash;
}

function checkIfArrayChangedAndUpdateHash(inputArray) {
  var hash = getHashOfArray(inputArray);
  if (hash !== sess.filepaths_hash) {
    sess.filepaths_hash = hash;
    return true;
  } else {
    return false;
  }
}

module.exports = {
  readFilesRecursively,
  readFilesSync,
  checkIfArrayChangedAndUpdateHash,
};
