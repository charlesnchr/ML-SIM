import React, { Component } from "react";
import "./App.css";

import {
    InputGroup
  } from "@blueprintjs/core";

import Grid from "@material-ui/core/Grid";
import TextField from "@material-ui/core/TextField";
import Button from "@material-ui/core/Button";
import ButtonGroup from "@material-ui/core/ButtonGroup";
import RefreshIcon from "@material-ui/icons/Refresh";
import blue from "@material-ui/core/colors/blue";
import grey from "@material-ui/core/colors/grey";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import Switch from "@material-ui/core/Switch";

import AddCircleIcon from "@material-ui/icons/AddCircle";
import RemoveCircleIcon from "@material-ui/icons/RemoveCircle";
import FolderOpenTwoToneIcon from "@material-ui/icons/FolderOpenTwoTone";

import Chip from "@material-ui/core/Chip";
import CollectionsIcon from "@material-ui/icons/Collections";

import Plugin_SIM from "./Plugin_SIM";
import Plugin_ERNet from "./Plugin_ERNet";

import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";

const { ipcRenderer } = window.require("electron");
const Store = window.require("electron-store");
const store = new Store();

const log = window.require("electron-log");
const remote = window.require("electron").remote;
var path = window.require("path");
var sess = require("./sess.js");

const mytheme = createMuiTheme({
  palette: {
    type: "dark",
  },
  overrides: {
    input: {
      fontSize: 10,
    },
    MuiInputLabel: {
      root: {
        "&$focused": {
          color: "#90caf9",
          fontWeight: "bold",
        },
      },
      focused: {},
    },
    MuiFormControl: {
      root: {
        marginTop: 10,
      },
    },
    MuiSelect: {
      root: {
        fontSize: 12,
      },
    },
    MuiMenuItem: {
      root: {
        fontSize: 12,
      },
    },
  },
});

class PluginComponents extends Component {
  constructor(props) {
    super(props);
    this.state = {
      plugins: store.get("plugins"),
      pluginCmps: [],
    };
  }

  updateSidePanelPlugins(plugins) {
    log.info("updated plugins", plugins);
    this.setState({ plugins: plugins, pluginCmps: this.components(plugins) });
  }

  components(plugins) {
    log.info("now rendering plugins");
    let activePluginCmps = [];
    plugins.forEach((activePlugin) => {
      [
        { name: "ML-SIM", cmp: <Plugin_SIM /> },
        { name: "ERNet", cmp: <Plugin_ERNet /> },
      ].forEach((_plugin) => {
        if (activePlugin === _plugin.name) {
          log.info("ADDING", _plugin.name);
          activePluginCmps.push(_plugin.cmp);
        }
      });
    });
    // let t = { name: "ML-SIM", cmp: <Plugin_SIM /> };
    //   return [t.cmp]
    return activePluginCmps;
  }

  componentDidMount() {
    sess.updateSidePanelPlugins = this.updateSidePanelPlugins.bind(this);
    this.updateSidePanelPlugins(this.state.plugins);
  }

  render() {
    return this.state.pluginCmps;
  }
}

class SidePanel extends Component {
  constructor(props) {
    super(props);

    this.state = {
      selectFile: props.selectFile,
      selectedFilepaths: [],
      resetResultImages: props.resetResultImages,
      useCloud: props.useCloud,
      useCloud_set: props.useCloud_set,
      folderEntry: props.folderEntry,
      selectedDir: null,
    };
  }

  componentDidMount() {
    sess.updateSidePanel = this.updateSidePanel.bind(this);
  }

  selectedDir_set(key) {
    if (this.state.selectedDir === key) {
      this.setState({ selectedDir: null });
      sess.selectedDir = null;
    } else {
      this.setState({ selectedDir: key });
      sess.selectedDir = key;
    }
  }

  updateSidePanel(selectedFilepaths) {
    sess.selectedFilepaths = selectedFilepaths;
    this.setState({ selectedFilepaths: selectedFilepaths });
  }

  addDir() {
    ipcRenderer.send("open-file-dialog");
  }

  removeDir() {
    this.setState({ selectedDir: null });
    sess.removeDir();
  }

  removeResults() {
    if (sess.removeResults) {
      // log.info("trying removeresults");
      sess.removeResults();
    }
  }

  folderEntry(key, dirpath, dirsize) {
    let label = path.basename(dirpath);
    if (label.length > 18) label = label.substr(0, 18) + "...";

    return (
      <Grid
        container
        key={key}
        justify="flex-start"
        alignItems="center"
        className={
          "folderEntry " +
          (this.state.selectedDir === key ? "selectedFolderEntry" : "")
        }
        onClick={this.selectedDir_set.bind(this, key)}
      >
        <Grid item>
          <FolderOpenTwoToneIcon
            style={{
              color: "#222222",
              fontSize: 18,
              marginRight: "10px",
              paddingTop: "2px",
            }}
          />
        </Grid>
        <Grid item style={{ width: "80%" }}>
          <Grid container justify="space-between">
            <Grid item>{label}</Grid>
            <Grid
              item
              style={{ fontStyle: "italic", color: blue[600], fontSize: 10 }}
            >
              {dirsize}
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    );
  }

  exportToFolder() {
    // log.info(this.state.selectedFilepaths);
    let selectedFilepaths = this.state.selectedFilepaths;
    ipcRenderer.send("sendToPython", {
      cmd: "exportToFolder",
      arg:
        remote.getGlobal("settings").exportdir +
        "\n" +
        selectedFilepaths,
    });
  }

  exportToZip() {
    // log.info(this.state.selectedFilepaths);
    let selectedFilepaths = this.state.selectedFilepaths;
    ipcRenderer.send("sendToPython", {
      cmd: "exportToZip",
      arg:
        remote.getGlobal("settings").exportdir +
        "\n" +
        selectedFilepaths,
    });
  }

  openInFolder() {
    // log.info(this.state.selectedFilepaths);
    let filename = this.state.selectedFilepaths[0];
    // let dir = path.dirname(filename)
    // const name = path.parse(filename).name;
    // log.info("running", 'explorer.exe /select, "' + filename + '"');
    window
      .require("child_process")
      .exec('explorer.exe /select,"' + filename + '"');
  }

  useCloud_handler(e) {
    this.setState({ useCloud: e.target.checked });
    this.state.useCloud_set(e.target.checked);
  }

  rescanFolders() {
    sess.thumbJobs = [];
    sess.readFolders(sess.displayedFolders);
  }

  updateExcludePattern(e) {
    sess.excludePattern = e.target.value;
    sess.readFolders(sess.displayedFolders);
  }

  updateIncludePattern(e) {
    sess.includePattern = e.target.value;
    sess.readFolders(sess.displayedFolders);
  }

  render() {
    // log.info("RERENDER sidepanel");

    const resetBtn = (
      <Button
        size="small"
        variant="outlined"
        color="secondary"
        id="reloadbtn"
        style={{ width: "100%", fontSize: 11 }}
        disabled={sess.showingResult ? false : true}
        onClick={this.removeResults.bind(this)}
      >
        Clear results
      </Button>
    );
    const cloudChk = (
      <FormControlLabel
        style={{ marginLeft: "10px", width: "100%", color: "#D9D9D9" }}
        label="Cloud compute"
        control={
          <Switch
            checked={this.state.useCloud}
            onChange={this.useCloud_handler.bind(this)}
            value="useCloud"
            color="primary"
            size="small"
          />
        }
      />
    );

    let folderEntries;

    if (sess.displayedFolders.length > 0) {
      folderEntries = sess.displayedFolders.map((dirpath, idx) => {
        let dirsize =
          sess.dirsizes && sess.dirsizes.length > idx
            ? sess.dirsizes[idx]
            : "N/A";
        return this.folderEntry(idx, dirpath, dirsize);
      });
    } else {
      folderEntries = "";
    }

    let conditionalRemoveCircleIcon =
      this.state.selectedDir !== null ? (
        <RemoveCircleIcon
          style={{ fontSize: 18, cursor: "pointer" }}
          onClick={this.removeDir.bind(this)}
        />
      ) : (
        []
      );

    return (
      <Grid container style={{ backgroundColor: "#292929" }}>
        {/* <Grid container>
              <Grid item>
                {grid}
              </Grid>
            </Grid> */}
        <Grid
          item
          style={{
            width: "100%",
            visibility:
              this.state.selectedFilepaths.length === 0 ? "hidden" : "visible",
          }}
        >
          <Chip
            icon={<CollectionsIcon />}
            label={this.state.selectedFilepaths.length + " selected"}
            color="secondary"
            onClick={sess.removeSelection}
            onDelete={sess.removeSelection}
            style={{ width: "100%" }}
          />
        </Grid>

        {/* FOLDERS */}

        <Grid item className="sidepanelSection">
          <Grid container justify="space-between">
            <Grid item>Folders</Grid>
            <Grid item>
              {conditionalRemoveCircleIcon}
              <AddCircleIcon
                style={{ fontSize: 18, cursor: "pointer" }}
                onClick={this.addDir.bind(this)}
              />
            </Grid>
          </Grid>
        </Grid>

        <Grid
          container
          direction="column"
          style={{
            padding: "20px 5px 20px 20px",
            backgroundColor: "#474747",
            color: "#D9D9D9",
            border: "1px solid black",
          }}
        >
          {folderEntries}
          <Grid item>
            <Button
              size="small"
              style={{
                fontSize: 11,
                marginLeft: 10,
                marginTop: 10,
                marginBottom: 20,
                backgroundColor: grey[700],
                color: grey[300],
              }}
              onClick={this.rescanFolders.bind(this)}
              variant="contained"
            >
              <RefreshIcon style={{ fontSize: 15 }} />
              &nbsp; Rescan folders
            </Button>
            <InputGroup
              className="bp3-dark"
              style={{width:185,marginBottom:10}}
              leftIcon="remove"
              placeholder="Exclude pattern"
              onChange={this.updateExcludePattern.bind(this)}
            />
            <InputGroup
              className="bp3-dark"
              style={{width:185}}
              leftIcon="add"
              placeholder="Include pattern"
              onChange={this.updateIncludePattern.bind(this)}
            />
          </Grid>
        </Grid>

        {/* Plugin elements */}

        <PluginComponents />

        {/* EXPORT */}

        <Grid item className="sidepanelSection">
          <Grid container justify="space-between">
            <Grid item style={{ paddingBottom: 5 }}>
              File management
            </Grid>
            <Grid item></Grid>
          </Grid>
        </Grid>

        <Grid
          container
          direction="column"
          style={{
            padding: "20px 20px 20px 20px",
            backgroundColor: "#474747",
            color: "#D9D9D9",
            border: "1px solid black",
          }}
        >
          <ButtonGroup
            orientation="vertical"
            variant="contained"
            aria-label="vertical outlined primary button group"
          >
            <Button
              style={{
                backgroundColor: grey[700],
                color: grey[300],
                fontSize: 12,
              }}
              size="small"
              onClick={this.exportToFolder.bind(this)}
            >
              Export to folder
            </Button>
            <Button
              style={{
                backgroundColor: grey[700],
                color: grey[300],
                fontSize: 12,
              }}
              size="small"
              onClick={this.exportToZip.bind(this)}
            >
              Export to zip
            </Button>
            <Button
              style={{
                backgroundColor: grey[700],
                color: grey[300],
                fontSize: 12,
              }}
              size="small"
              onClick={this.openInFolder.bind(this)}
            >
              Show in folder
            </Button>
          </ButtonGroup>
        </Grid>
      </Grid>
    );
  }
}

export default SidePanel;
