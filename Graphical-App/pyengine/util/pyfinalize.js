var fs = require("fs");
var path = require("path");
var crypto = require("crypto");

const getHash = (filename, callback) => {
  // the file you want to get the hash
  var fd = fs.createReadStream(filename);
  var hash = crypto.createHash("sha1");
  hash.setEncoding("hex");

  fd.on("end", function () {
    hash.end();
    let hashval = hash.read();
    callback(hashval);
  });

  // read all file and pipe it (write it) to the hash object
  fd.pipe(hash);
}

let engine_dir = "."
let engine_zip = 'engine.mlsim'
let version = require('../package.json').version;
let data = { version: version }

console.time('reading model hash')
getHash(path.join(engine_dir, engine_zip), (engine_hash) => {
  console.timeEnd('reading model hash')
  console.log(engine_zip, engine_hash)
  data.engine = { url: engine_zip, hash: engine_hash }
  data.models = []

  let modelsdir = path.join(engine_dir, 'models')
  let models = fs.readdirSync(modelsdir)

  models.forEach(model => {
    getHash(path.join(modelsdir, model), (model_hash) => {
      console.log(model, model_hash)
      data.models.push({ url: model, hash: model_hash })
      if (data.models.length === models.length) {
        let datajson = JSON.stringify(data)
        fs.writeFileSync(path.join(engine_dir, 'latest.json'), datajson);
      }
    })
  });
})
