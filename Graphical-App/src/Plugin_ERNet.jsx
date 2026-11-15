import React, { Component } from "react";
import "./App.css";

import Grid from "@material-ui/core/Grid";
import CalcStatus from "./CalcStatus";

import ExtensionIcon from "@material-ui/icons/Extension";
import Button from "@material-ui/core/Button";
import TextField from "@material-ui/core/TextField";
import FormControl from "@material-ui/core/FormControl";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import Box from "@material-ui/core/Box";
import InputLabel from "@material-ui/core/InputLabel";
import Checkbox from "@material-ui/core/Checkbox";
import MenuItem from "@material-ui/core/MenuItem";
import Select from "@material-ui/core/Select";
import grey from "@material-ui/core/colors/grey";
import PlayArrowIcon from "@material-ui/icons/PlayArrow";

import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";

const { ipcRenderer } = window.require("electron");
const log = window.require("electron-log");
const remote = window.require("electron").remote;
var sess = require("./sess.js");

const mytheme = createMuiTheme({
  palette: {
    type: "dark",
  },
  overrides: {
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

export default class Plugin_SIM extends Component {
  constructor(props) {
    super(props);
    this.state = {
      panelTitle: "ERNet",
      wekaColours: true,
      tubuleSheetStats: true,
      graphMetrics: true,
      saveInOriginalFolders: true,
    };
  }

  toggle_wekaColours() {
    this.setState({ wekaColours: !this.state.wekaColours });
  }

  toggle_tubuleSheetStats() {
    this.setState({ tubuleSheetStats: !this.state.tubuleSheetStats });
  }

  toggle_graphMetrics() {
    this.setState({ graphMetrics: !this.state.graphMetrics });
  }

  toggle_saveInOriginalFolders() {
    this.setState({ saveInOriginalFolders: !this.state.saveInOriginalFolders });
  }

  segmentImages() {
    let filepaths;
    if (sess.selectedFilepaths.length > 0)
      filepaths = sess.selectedFilepaths;
    else filepaths = sess.filepaths;

    log.info("sending", filepaths);

    ipcRenderer.send("sendToPython", {
      cmd: "Plugin_ERNet",
      filepaths: filepaths,
      arg:
        remote.getGlobal("settings").exportdir +
        "\n" +
        this.state.wekaColours +
        "\n" +
        this.state.tubuleSheetStats +
        "\n" +
        this.state.graphMetrics +
        "\n" +
        this.state.saveInOriginalFolders,
    });
  }

  render() {
    return (
      <>
        <Grid item className="sidepanelSection">
          <Grid container justify="space-between">
            <Grid item style={{ paddingBottom: 5 }}>
              <ExtensionIcon
                style={{
                  fontSize: 16,
                  color: "rgb(51, 204, 51)",
                  marginRight: "10px",
                  marginBottom: "-3px",
                }}
              />
              {this.state.panelTitle}
            </Grid>
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
          <Grid
            key={"grid"}
            direction="column"
            container
            spacing={1}
            justify="flex-start"
            style={{ padding: "0 6px 0 3px" }}
          >
            <Grid item className="calcStatus">
              <CalcStatus />
            </Grid>

            <Grid item>
              <Button
                size="large"
                style={{
                  fontSize: 11,
                  marginLeft: 10,
                  marginRight: 20,
                  backgroundColor: grey[700],
                  color: grey[300],
                }}
                onClick={this.segmentImages.bind(this)}
                variant="contained"
              >
                <PlayArrowIcon style={{ fontSize: 15 }} />
                &nbsp; Perform segmentation
              </Button>
            </Grid>
            <Grid item>
              <ThemeProvider theme={mytheme}>
                <TextField
                  id="standard-input"
                  label="Image tile size (pixels)"
                  defaultValue="1000"
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      color="primary"
                      checked={this.state.wekaColours}
                      onChange={this.toggle_wekaColours.bind(this)}
                      name="wekaColours"
                    />
                  }
                  label={
                    <Box component="div" fontSize={14}>
                      WEKA colours
                    </Box>
                  }
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      color="primary"
                      checked={this.state.tubuleSheetStats}
                      onChange={this.toggle_tubuleSheetStats.bind(this)}
                      name="wekaColours"
                    />
                  }
                  label={
                    <Box component="div" fontSize={14}>
                      Tubule/sheet stats
                    </Box>
                  }
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      color="primary"
                      checked={this.state.graphMetrics}
                      onChange={this.toggle_graphMetrics.bind(this)}
                      name="graphMetrics"
                    />
                  }
                  label={
                    <Box component="div" fontSize={14}>
                      Generate graph metrics
                    </Box>
                  }
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      color="primary"
                      checked={this.state.saveInOriginalFolders}
                      onChange={this.toggle_saveInOriginalFolders.bind(this)}
                      name="wekaColours"
                    />
                  }
                  label={
                    <Box component="div" fontSize={14}>
                      Save output in original folders
                    </Box>
                  }
                />
                <FormControl className="modelSelect">
                  <InputLabel>Selected model</InputLabel>
                  <Select defaultValue="mdl1">
                    <MenuItem value="mdl1">
                      Three-colour tubuleSheet_4.pth
                    </MenuItem>
                    <MenuItem value="mdl2">Randomised RCAN - 3x5.pth</MenuItem>
                    <MenuItem value="mdl3">
                      My optimised model - 3x5.pth
                    </MenuItem>
                  </Select>
                </FormControl>
              </ThemeProvider>
            </Grid>
          </Grid>
        </Grid>
      </>
    );
  }
}
