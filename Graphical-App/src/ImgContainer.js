import React, { Component } from "react";
import blue from "@material-ui/core/colors/blue";
import Box from "@material-ui/core/Box";
import Img from "./Img";

const { ipcRenderer } = window.require("electron");
const log = window.require("electron-log");
const shell = window.require("electron").shell;

var sess = require("./sess.js");

function calculateGeometry(filepaths, imgsize) {
  var nimgs = filepaths.length;

  var screenheight = window.innerHeight - sess.topOffset;
  var screenwidth = window.innerWidth - sess.sidePanelWidth;
  var Nrow = parseInt(screenheight / (imgsize + 2 * 5));
  var Ncol = parseInt(screenwidth / (imgsize + 2 * 5));
  var Nblock = Nrow * Ncol;
  var blockheight = (imgsize + 2 * 5) * Nrow;
  var baseoffset = sess.topOffset;
  let gridheight;
  if (nimgs === 0) {
    gridheight = 500;
  } else {
    gridheight = parseInt(nimgs / Nblock) * blockheight;
  }
  log.info(
    "calculated geometry",
    nimgs,
    sess.topOffset,
    imgsize,
    Nrow,
    Ncol,
    Nblock,
    window.innerWidth,
    window.innerHeight
  );
  return {
    Nblock: Nblock,
    blockheight: blockheight,
    baseoffset: baseoffset,
    gridheight: gridheight,
  };
}

class ImgContainer extends Component {
  constructor(props) {
    super(props);

    var startidx = 0;
    var g = calculateGeometry(sess.filepaths, sess.imgsize);
    props.gridheight_set(g.gridheight);

    this.state = {
      startidx: startidx,
      gridheight_set: props.gridheight_set,
      gridheight: 0,
      Nblock: g.Nblock,
      blockheight: g.blockheight,
      baseoffset: g.baseoffset,
      resultImages: null,
      selectFile: props.selectFile,
      selectedFilepaths: [],
      storedGeometry: {},
      filepaths: []
    };

    // log.info("NOW SETTING UPDATE");
    sess.updateGeometry = this.updateGeometry.bind(this);
    sess.updateFilepaths = this.updateGeometry.bind(this);
    sess.removeResults = this.removeResults.bind(this);
    sess.revertResults = this.revertResults.bind(this);
    sess.recoverResults = this.recoverResults.bind(this);
    sess.imgsizeRerender = this.imgsizeRerender.bind(this);
    sess.removeSelection = this.removeSelection.bind(this);
    sess.toggleSelectedFilepath = this.toggleSelectedFilepath.bind(this);
  }

  toggleSelectedFilepath(filepath) {
    let filepaths = this.state.selectedFilepaths;

    let idx = filepaths.indexOf(filepath);
    if (idx > -1) {
      filepaths.splice(idx, 1);
    } else {
      filepaths.push(filepath);
    }

    this.setState({ selectedFilepaths: filepaths });
    sess.updateSidePanel(filepaths);
  }

  checkForServerActive() {
    if (!sess.serverActive) {
      ipcRenderer.send("isServerActive");
      setTimeout(
        function() {
          this.checkForServerActive();
        }.bind(this),
        500
      );
    }
  }

  componentDidMount() {
    /******************
     * EVENT HANDLERS *
     ******************/
    ipcRenderer.on("serverActive", (event) => {
      log.info("serverActive now active");
      sess.serverActive = true;
      sess.readFolders(sess.displayedFolders);
    });

    this.checkForServerActive();

    ipcRenderer.on("ReceivedResults", (event, result) => {
      if (result.length === 0) {
        log.info("empty result");
        ipcRenderer.send("emptyResultBox");
        return;
      }

      let filepaths = result.split("\n");

      sess.showingResult = true;
      sess.resultsDepth = 1;
      sess.resultsBuffer = [];
      sess.resultsBuffer.push(filepaths);
      // if (sess.resultsDepth === 0) sess.resultsBuffer = [];
      // sess.resultsDepth++;
      // if (sess.resultsBuffer.length < sess.resultsDepth - 1)
      //   sess.resultsBuffer.push(filepaths);
      // else {
      //   if (sess.resultsBuffer.length > sess.resultsDepth)
      //     sess.resultsBuffer = sess.resultsBuffer.slice(0, sess.resultsDepth);
      //   sess.resultsBuffer[sess.resultsDepth - 1] = filepaths;
      // }

      log.info("UPDATED resultsbuffer", sess.resultsBuffer);
      this.updateGeometry(filepaths);
      this.setState({ selectedFilepaths: [], resultImages: filepaths });
    });

    window.addEventListener("scroll", () => {
      if (!window.updateblocks) return;
      var blocksAboveLimit = parseInt(
        (window.pageYOffset - (window.innerHeight - sess.topOffset) / 3) /
          this.state.blockheight
      );
      if (blocksAboveLimit < 0) blocksAboveLimit = 0;
      var baseoffset =
        blocksAboveLimit * this.state.blockheight + sess.topOffset;
      var startidx = blocksAboveLimit * this.state.Nblock;
      this.setState({ baseoffset: baseoffset, startidx: startidx });
    });

    window.addEventListener("resize", () => {
      let g = calculateGeometry(sess.filepaths, sess.imgsize);
      var blocksAboveLimit = parseInt(
        (window.pageYOffset - (window.innerHeight - sess.topOffset) / 3) /
          g.blockheight
      );
      if (blocksAboveLimit < 0) blocksAboveLimit = 0;
      var baseoffset = blocksAboveLimit * g.blockheight + sess.topOffset;
      var startidx = blocksAboveLimit * g.Nblock;

      this.state.gridheight_set(g.gridheight);
      this.setState({
        baseoffset: baseoffset,
        startidx: startidx,
        Nblock: g.Nblock,
        blockheight: g.blockheight,
        gridheight: g.gridheight,
      });
    });
  }

  imgsizeRerender() {
    let g = calculateGeometry(sess.filepaths, sess.imgsize);
    var blocksAboveLimit = parseInt(
      (window.pageYOffset - (window.innerHeight - sess.topOffset) / 3) /
        g.blockheight
    );
    if (blocksAboveLimit < 0) blocksAboveLimit = 0;
    var baseoffset = blocksAboveLimit * g.blockheight + sess.topOffset;
    var startidx = blocksAboveLimit * g.Nblock;

    this.state.gridheight_set(g.gridheight);
    this.setState({
      baseoffset: baseoffset,
      startidx: startidx,
      Nblock: g.Nblock,
      blockheight: g.blockheight,
      gridheight: g.gridheight,
    });
  }


  updateGeometry(filepaths) {
    let g,
      storedGeometry = this.state.storedGeometry;

    // if displaying results for the first time, save display geometry
    if (sess.showingResult) {
      if (Object.keys(this.state.storedGeometry).length === 0) {
        storedGeometry["Nblock"] = this.state.Nblock;
        storedGeometry["blockheight"] = this.state.blockheight;
        storedGeometry["gridheight"] = this.state.gridheight;
        storedGeometry["startidx"] = this.state.startidx;
        storedGeometry["baseoffset"] = this.state.baseoffset;
        storedGeometry["pageYOffset"] = window.pageYOffset;
        storedGeometry["filepaths"] = sess.filepaths;
      }
      sess.showHideSelect(false);
      sess.updateSidePanel([]);
      sess.hideBackdrop();
    }

    // new geometry
    if (filepaths) {
      g = calculateGeometry(filepaths, sess.imgsize);
    } else {
      g = calculateGeometry(sess.filepaths, sess.imgsize);
    }

    sess.filepaths = filepaths;

    // log.info("updating geometry", sess.imgsize, g);
    this.state.gridheight_set(g.gridheight);
    this.setState({
      Nblock: g.Nblock,
      blockheight: g.blockheight,
      gridheight: g.gridheight,
      storedGeometry: storedGeometry,
      filepaths: sess.filepaths
    });
    return g;
  }

  removeResults() {
    // sess.readFolders(sess.displayedFolders);
    // this.resetGeometry()
    sess.showingResult = false;
    sess.showHideSelect(true);
    sess.resultsDepth = 0;
    let storedGeometry = this.state.storedGeometry;
    this.setState({
      Nblock: storedGeometry["Nblock"],
      blockheight: storedGeometry["blockheight"],
      gridheight: storedGeometry["gridheight"],
      startidx: storedGeometry["startidx"],
      baseoffset: storedGeometry["baseoffset"],
      storedGeometry: {},
      selectedFilepaths: [],
      resultImages: null,
    });

    sess.pageYOffset = storedGeometry["pageYOffset"];
    this.state.gridheight_set(storedGeometry["gridheight"]);
    // window.scrollTo(0,storedGeometry['pageYOffset'])
    sess.filepaths = storedGeometry["filepaths"];
    sess.updateSidePanel([]);

    // log.info("now changing imgcontainer state");
  }

  revertResults() {
    log.info(
      "now here TRYING to revert with",
      sess.resultsDepth,
      sess.resultsBuffer.length
    );
    if (sess.resultsDepth < 2) {
      this.removeResults();
    } else {
      sess.resultsDepth--;
      var filepaths = sess.resultsBuffer[sess.resultsDepth - 1];
      this.updateGeometry(filepaths);
      this.setState({ selectedFilepaths: [], resultImages: filepaths });
    }
  }

  recoverResults() {
    if (sess.resultsBuffer.length > sess.resultsDepth) {
      log.info(
        "now here TRYING to recover with",
        sess.resultsDepth,
        sess.resultsBuffer.length
      );
      sess.showingResult = true;
      var filepaths = sess.resultsBuffer[sess.resultsDepth];
      sess.resultsDepth++;
      this.updateGeometry(filepaths);
      this.setState({ selectedFilepaths: [], resultImages: filepaths });
    }
  }

  removeSelection() {
    sess.updateSidePanel([]);
    this.setState({ selectedFilepaths: [] });
  }

  imgGrid(filepaths, startidx, Nblock, offset) {
    log.info("remaking imgGrid",startidx);
    let filepaths_displayed = filepaths.slice(startidx, startidx + Nblock);
    let selectedFilepaths = this.state.selectedFilepaths;
    const imgdivs = filepaths_displayed.map((filepath, idx) => {
      const selectedBool = selectedFilepaths.indexOf(filepath) > -1;
      return <Img selectedBool={selectedBool} filepath={filepath} key={"img-" + filepath} />;
    });
    log.info("KEY","imgGrid-" +
    sess.filepaths_hash +
    "-" +
    this.state.resultImages +
    "-" +
    startidx +
    "-" +
    sess.sortBy  );

    return (
      <Box
        style={{ position: "absolute", top: offset }}
        key={
          "imgGrid-" +
          sess.filepaths_hash +
          "-" +
          this.state.resultImages +
          "-" +
          startidx +
          "-" +
          sess.sortBy
        }
        id="imgcontainer"
      >
        {imgdivs}
      </Box>
    );
  }

  downloadImages() {
    shell.openExternal("https://ML-SIM.com/images");
  }

  render() {
    if (!this.state.resultImages && sess.displayedFolders.length === 0) {
      return (
        <div
          style={{
            marginLeft: 50,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            fontSize: 25,
            width: 400,
            fontFamily: "Roboto",
          }}
        >
          <p>Choose a folder to begin.</p>
          <p style={{ fontSize: 15 }}>
            {" "}
            You can also{" "}
            <span
              style={{ cursor: "pointer", color: blue[800] }}
              onClick={this.downloadImages.bind(this)}
            >
              try our official library
            </span>{" "}
            of test images to see how everything works.{" "}
          </p>{" "}
        </div>
      );
    }

    let filepaths,
      imgs = <div></div>;

    if (this.state.resultImages) {
      // log.info("RERENDER results");
      filepaths = this.state.resultImages;
    } else if (sess.displayedFolders.length > 0) {
      // log.info("SHOWING IMAGES ", sess.filepaths.length);
      filepaths = sess.filepaths;
    }
    var startidx = this.state.startidx;
    var Nblock = this.state.Nblock;
    var blockheight = this.state.blockheight;
    var baseoffset = this.state.baseoffset;
    // log.info(
    //   "RERENDER Imgcontainer: ",
    //   startidx,
    //   Nblock,
    //   blockheight,
    //   baseoffset,
    //   sess.showingResult,
    //   sess.resultImages
    // );
    sess.thumbQueue = [];
    log.info("redrawing",filepaths[0]);
    imgs = [
      this.imgGrid(filepaths, startidx, Nblock, baseoffset),
      this.imgGrid(
        filepaths,
        startidx + Nblock,
        Nblock,
        baseoffset + blockheight
      ),
      this.imgGrid(
        filepaths,
        startidx + 2 * Nblock,
        Nblock,
        baseoffset + 2 * blockheight
      ),
    ];

    return <div justify="flex-start">{imgs}</div>;
  }
}

export default ImgContainer;
