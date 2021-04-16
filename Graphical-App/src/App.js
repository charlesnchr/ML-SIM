import React, { Component } from "react";

import Grid from "@material-ui/core/Grid";

import {
  Button as BPButton,
  // ButtonGroup as BPButtonGroup,
  Icon,
  InputGroup,
  Intent,
  Tooltip,
} from "@blueprintjs/core";
// import { Colors } from "@blueprintjs/core";
import SettingsMenu from "./SettingsMenu";

import ImgContainer from "./ImgContainer";
import ImgsizeSlider from "./ImgsizeSlider";
import SidePanel from "./SidePanel";
import AppUpdate from "./AppUpdate";
import AboutMenu from "./AboutMenu";
import WindowMenuRender from "./WindowMenu";

import logo from "./logo.png";

import Backdrop from "@material-ui/core/Backdrop";
import CircularProgress from "@material-ui/core/CircularProgress";

import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";


const { ipcRenderer } = window.require("electron");
const shell = window.require("electron").shell;
const remote = window.require("electron").remote;

const app = remote.app;

const Store = window.require("electron-store");
const store = new Store();

let plugins = store.get("plugins");
if (plugins == null) {
  store.set("plugins", ["ML-SIM"]);
}

const log = window.require("electron-log");
const isDev = window.require("electron-is-dev");

var sess = require("./sess.js");

var FileUtils = require("./FileUtils.js");

if (isDev && false) sess.baseurl = "http://localhost:5000"; // dev of authentication etc.

window.updateblocks = true;


const theme = createMuiTheme({
  palette: {},
  typography: {
    fontFamily: "Roboto",
    body1: {
      fontFamily: "Roboto",
      fontSize: 14,
    },
    body2: {
      fontFamily: "Roboto",
      fontSize: 10,
    },
    div: {
      fontFamily: "Roboto",
    },
  },
});




ipcRenderer.on("selected-file", (event, filepath) => {
  ipcRenderer.send("sendToPython", {
    cmd: "Reconstruct",
    arg: remote.getGlobal("settings").exportdir + "\n" + filepath.join("\n"),
  });
});



class App extends Component {
  constructor(props) {
    super(props);

    let displayedFolders = store.get("displayedFolders");
    if (!displayedFolders) displayedFolders = [];
    this.state = {
      imgsize: sess.imgsize,
      resultImages: null,
      displayedFolders: displayedFolders,
      selectedDir: null,
      useCloud: store.get("useCloud"),
      gridheight: 500,
      width: null,
      height: null,
      render_status: 0,
      appWidth: window.innerWidth,
      settingsOpen: false,
      aboutOpen: false,
      desktop_fingerprint: sess.desktop_fingerprint,
      sortBy: null,
      showBackdrop: false,
      engineStatus: "unstarted",
    };

    sess.displayedFolders = displayedFolders;
    sess.dirsize = store.get("dirsizes");
    sess.selectedDir = null;
    sess.filepaths = [];
    sess.dirsizes = [];
    sess.readFolders = this.readFolders.bind(this);
    sess.showBackdrop = this.showBackdrop.bind(this);
    sess.hideBackdrop = this.hideBackdrop.bind(this);
    if (!isDev) AppUpdate(); // update service
  }

  showBackdrop() {
    this.setState({ showBackdrop: true });
  }

  hideBackdrop() {
    this.setState({ showBackdrop: false });
  }

  selectFile() {
    ipcRenderer.send("open-singlefile-dialog");
  }

  imgsizeHandler(value) {
    if (value !== this.state.imgsize) {
      sess.imgsize = value;
      this.setState({ imgsize: value });
      if (sess.imgsizeRerender) sess.imgsizeRerender();
    }
  }

  loggedIn_set(loggedIn, firstname, lastname, emailaddr, token) {
    this.setState({
      loggedIn: loggedIn,
      firstname: firstname,
      lastname: lastname,
      emailaddr: emailaddr,
    });
  }

  setWidth() {
    if (this.state.appWidth !== window.innerWidth) {
      this.setState({ appWidth: window.innerWidth });
    }
  }

  logout() {
    this.setState({ loggedIn: false });
    store.set("loggedIn", false);
    store.set("loggedIn_firstname", null);
    store.set("loggedIn_lastname", null);
    store.set("loggedIn_emailaddr", null);
    WindowMenuRender();
  }

  componentDidMount() {
    ipcRenderer.on("EngineStatus", (event, res) => {
      if (res === "a") {
        if (this.state.engineStatus === "unstarted") {
          this.setState({ engineStatus: "active" });
        } else {
          // if installation has just finished
          this.setState({ engineStatus: "recently-active" });
          setTimeout(() => {
            this.setState({ engineStatus: "active" });
          }, 5000);
        }
        log.info("Engine connection from renderer");
        if (sess.filepaths != null && sess.filepaths_hash !== "") {
          log.info("Filepaths ready - can send to engine");
          // ipcRenderer.send("sendToPython", {
          //   cmd: "calcFeatures",
          //   filepaths_hash: sess.filepaths_hash,
          //   arg: sess.filepaths,
          //   postIndexOp: "none",
          // });
        } else {
          // readfolders will initiate loading
          log.info("Filepaths not ready yet");
        }
      } else {
        // engine not ready, re-query
        if (res === "e") {
          this.setState({ engineStatus: "Installing engine.." });
          log.info("Renderer: extraction ongoing");
        } else if (res[0] === "d") {
          let dl_status = res.split(",")[1];
          let downloadMsg = "Downloading required files.. " + dl_status + " %";
          this.setState({ engineStatus: downloadMsg });
          log.info("Renderer: Dowloading, status", dl_status);
        } else {
          // log.info("Renderer: Something else is stalling");
          if (this.state.engineStatus !== "unstarted") {
            this.setState({ engineStatus: "Preparing.." });
          }
        }
        setTimeout(() => {
          ipcRenderer.send("EngineStatus");
        }, 100);
      }
    });

    ipcRenderer.send("EngineStatus");

    // EVENT HANDLERS

    ipcRenderer.on("selected-directory", (event, dirpath) => {
      // Add directory
      let displayedFolders = this.state.displayedFolders;
      displayedFolders.push(dirpath);

      this.readFolders(displayedFolders);
    });

    window.addEventListener("resize", this.setWidth.bind(this));

    // BINDINGS

    sess.logout = this.logout.bind(this);
    sess.removeDir = this.removeDir.bind(this);
    sess.loggedIn_set = this.loggedIn_set.bind(this);
    window.setRenderState = function(val) {
      if (this.state.render_status !== val)
        this.setState({ render_status: val });
    }.bind(this);

    sess.settingsHandler = this.settingsOpen.bind(this);
    sess.aboutHandler = this.aboutOpen.bind(this);

    WindowMenuRender();
  }

  componentDidUpdate() {
    if (sess.pageYOffset > 0 && !sess.showingResult) {
      // log.info("first part");
      window.scrollTo(0, sess.pageYOffset);
      sess.pageYOffset = 0;
    } else if (sess.showingResult) {
      window.scrollTo(0, 0);
    }
  }

  readFolders(displayedFolders, sortBy) {
    log.info("Reading folders", displayedFolders);
    if (sortBy == null) sortBy = this.state.sortBy;
    this.setState({ displayedFolders: displayedFolders, sortBy: sortBy });
    if (true) {
      FileUtils.readFilesRecursively(
        displayedFolders,
        [".tif", ".TIF", ".tiff", ".TIFF",".png",".PNG"],
        sortBy,
        this.readFoldersCallback.bind(this)
      );
    } else {
      var filedata = FileUtils.readFilesSync(
        displayedFolders,
        [".tif", ".TIF", ".tiff", ".TIFF",".png",".PNG"],
        sortBy
      );
      this.readFoldersCallback(filedata);
    }

    sess.displayedFolders = displayedFolders;
    store.set("displayedFolders", displayedFolders);
  }

  readFoldersCallback(filedata) {
    let g;
    if (FileUtils.checkIfArrayChangedAndUpdateHash(filedata.filepaths)) {
      log.info("Filepaths change detected", filedata.dirsizes);

      sess.filepaths = filedata.filepaths;
      sess.dirsizes = filedata.dirsizes;
      g = sess.updateGeometry(sess.filepaths);
      this.setState({ gridheight: g.gridheight });
    } else if (sess.resort) {
      log.info(sess.filepaths[0],'\nVERSUS\n',filedata.filepaths[0]);
      sess.filepaths = filedata.filepaths;
      sess.dirsizes = filedata.dirsizes;
      log.info(sess.filepaths[0],'\nVERSUS\n',filedata.filepaths[0]);
      g = sess.updateGeometry(sess.filepaths);
      this.setState({ gridheight: g.gridheight });
      sess.resort = false;
    }
  }

  removeDir() {
    if (sess.selectedDir !== null) {
      let displayedFolders = this.state.displayedFolders;
      if (displayedFolders.length === 0) return;
      displayedFolders = displayedFolders.filter((dirpath, idx) => {
        if (idx === sess.selectedDir) return false;
        return true;
      });
      sess.selectedDir = null;

      this.readFolders(displayedFolders);
    }
  }

  renewSubscription() {
    shell.openExternal(sess.baseurl + "/profile");
  }

  // state handling
  displayedFolders_set(displayedFolders) {
    this.setState({ displayedFolders: displayedFolders });
  }
  gridheight_set(gridheight) {
    // log.info("SETTING gridheight", gridheight);
    this.setState({ gridheight: gridheight });
  }

  useCloud_get() {
    return this.state.useCloud;
  }
  useCloud_set(val) {
    store.set("useCloud", val);
    const checked = val ? 1 : 0;
    ipcRenderer.send("sendToPython", { cmd: "SetUseCloud", arg: checked });
    this.setState({ useCloud: val });
  }

  settingsOpen() {
    this.setState({ settingsOpen: true });
  }
  settingsClose() {
    this.setState({ settingsOpen: false });
  }
  aboutOpen() {
    this.setState({ aboutOpen: true });
  }
  aboutClose() {
    this.setState({ aboutOpen: false });
  }

  engineStatus() {
    let estat = this.state.engineStatus;
    log.info("inside engine status", estat);
    if (estat === "active" || estat === "unstarted") {
      return "";
    } else if (estat === "recently-active") {
      return (
        <div
          style={{
            width: "100%",
            height: "auto",
            position: "fixed",
            backgroundColor: "green",
            zIndex: 1000,
            bottom: 0,
            padding: 10,
            fontWeight: "bold",
            marginLeft: sess.sidePanelWidth,
          }}
        >
          The Python engine is now installed! &nbsp;
          <span style={{ fontWeight: "normal" }}>
            Processing of your images can begin.
          </span>
        </div>
      );
    } else {
      var progress = (
        <CircularProgress
          style={{ marginLeft: 15, marginBottom: -3 }}
          size={18}
        />
      );
      return (
        <div
          style={{
            width: "100%",
            height: "auto",
            position: "fixed",
            backgroundColor: "orange",
            zIndex: 1000,
            bottom: 0,
            padding: 10,
            fontWeight: "bold",
            marginLeft: sess.sidePanelWidth,
          }}
        >
          Engine not ready:{" "}
          <span style={{ fontWeight: "normal" }}>{estat}</span>
          {estat === "Installing engine.." ? progress : ""}
        </div>
      );
    }
  }

  render() {
    const appHeader = (
      <div
        className="App-header draggable"
        style={{
          position: "fixed",
          width: this.state.appWidth,
          zIndex: 100,
          height: sess.headerHeight,
        }}
      >
        <Grid container alignItems="center" justify="space-between" spacing={2}>
          <Grid item style={{ width: sess.sidePanelWidth }}>
            <img src={logo} alt="logo" height={40} style={{marginTop:12}} />
          </Grid>
          <Grid item style={{ marginBottom: 5 }}>
            <BPButton
              large={true}
              disabled={!sess.showingResult}
              minimal={true}
              intent={Intent.PRIMARY}
              onClick={sess.revertResults}
              style={{ marginRight: 5 }}
            >
              <Icon icon="arrow-left" iconSize={20} />
            </BPButton>
            <BPButton
              large={true}
              disabled={sess.resultsBuffer.length <= sess.resultsDepth}
              minimal={true}
              intent={Intent.PRIMARY}
              onClick={sess.recoverResults}
            >
              <Icon icon="arrow-right" iconSize={20} />
            </BPButton>
          </Grid>
          {/* <Grid item >
                            <UpdateStatus/>
                          </Grid> */}

          <Grid item xs style={{ paddingRight: 50, paddingBottom: 10 }}>
            <ImgsizeSlider
              imgsize={this.state.imgsize}
              handler={this.imgsizeHandler.bind(this)}
            />
          </Grid>
        </Grid>
      </div>
    );

    return (
      <ThemeProvider theme={theme}>
        <div className="App">
          <SettingsMenu
            open={this.state.settingsOpen}
            onClose={this.settingsClose.bind(this)}
            exportdir={remote.getGlobal("settings").exportdir}
            cachedir={remote.getGlobal("settings").cachedir}
          />
          <AboutMenu
            open={this.state.aboutOpen}
            onClose={this.aboutClose.bind(this)}
            version={app.getVersion()}
          />
          {appHeader}
          <Backdrop
            style={{ zIndex: 100, color: "#fff" }}
            open={this.state.showBackdrop}
            onClick={sess.hideBackdrop}
          >
            <CircularProgress color="inherit" />
          </Backdrop>

          <Grid container justify="flex-start">
            <Grid
              item
              style={{
                overflowX: "hidden",
                flex: "0 0 " + sess.sidePanelWidth,
                backgroundColor: "#292929",
                maxWidth: sess.sidePanelWidth,
                position: "fixed",
                height: window.innerHeight,
                paddingTop: 70,
              }}
            >
              <SidePanel
                selectFile={this.selectFile}
                resetResultImages={this.resetResultImages}
                useCloud={this.state.useCloud}
                useCloud_set={this.useCloud_set.bind(this)}
              />
            </Grid>
            <Grid item xs>
              <div
                style={{
                  display: "flex",
                  height: this.state.gridheight,
                  paddingLeft: sess.sidePanelWidth + 15,
                }}
              >
                <ImgContainer
                  resultImages={this.state.resultImages}
                  gridheight_set={this.gridheight_set.bind(this)}
                  selectFile={this.selectFile}
                />
                {/* {this.imgContainer()} */}
              </div>
            </Grid>
          </Grid>
          {/* <div style={{width:'230px',height:'700px',position:'fixed',left:0,top:0,backgroundColor:'black'}}></div> */}
          {this.engineStatus()}
        </div>
      </ThemeProvider>
    );
  }
}

export default App;
