import React, { Component } from "react";
import Grid from "@material-ui/core/Grid";
import CalcStatus from "./CalcStatus";

import ExtensionIcon from '@material-ui/icons/Extension';
import Button from "@material-ui/core/Button";
import Switch from "@material-ui/core/Switch";
import FormControlLabel from "@material-ui/core/FormControlLabel";
import FormControl from "@material-ui/core/FormControl";
import InputLabel from "@material-ui/core/InputLabel";
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
    MuiFormControlLabel: {
      label:{
        fontSize: 12
      }
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

export default class Plugin_SIM extends Component {
  constructor(props) {
    super(props);
    this.state = {
        panelTitle: 'ML-SIM',
        MM_integration: false
    }
  }
  
  MM_integration_handler() {
    log.info('Sending activation signal for MM_integration to Python engine')
    let MM_integration = !this.state.MM_integration;
    let MM_integration_str = MM_integration ? 'on' : 'off';
    let port = 5003;

    ipcRenderer.send("sendToPython", {
      cmd: "Plugin_MLSIM_MM_integration",
      arg: MM_integration_str + "\n" + port
    });
    this.setState({MM_integration:MM_integration});
  }

  reconstructImages() {
    let filepaths;
    if (sess.selectedFilepaths.length > 0)
      filepaths = sess.selectedFilepaths;
    else filepaths = sess.filepaths;

      // ipcRenderer.send("open-singlefile-dialog");
    // ipcRenderer.send("sendToPython", {
    //   cmd: "Reconstruct",
    //   arg:
    //     remote.getGlobal("settings").exportdir +
    //     "\n" +
    //     args.filepath.join("\n"),
    // });
    // }
    
    ipcRenderer.send("sendToPython", {
      cmd: "Plugin_MLSIM",
      filepaths: filepaths,
      arg:
        remote.getGlobal("settings").exportdir
    });
  }

  render() {
    return (
      <>
        <Grid item className="sidepanelSection">
          <Grid container justify="space-between">
            <Grid item style={{ paddingBottom: 5 }}>
              <ExtensionIcon style={{
              fontSize: 16,
              color: "rgb(51, 204, 51)",
              marginRight: "10px",
              marginBottom: "-3px"
            }}/>
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
                  backgroundColor: grey[700],
                  color: grey[300],
                }}
                onClick={this.reconstructImages.bind(this)}
                variant="contained"
              >
                <PlayArrowIcon style={{ fontSize: 15 }} />
                &nbsp; Run ML-SIM
              </Button>
            </Grid>
            <Grid item>
              <ThemeProvider theme={mytheme}>
                <FormControl className="modelSelect">
                  <InputLabel>Selected model</InputLabel>
                  <Select defaultValue="mdl1">
                    <MenuItem value="mdl1">Randomised RCAN - 3x3.pth</MenuItem>
                    <MenuItem value="mdl2">Randomised RCAN - 3x5.pth</MenuItem>
                    <MenuItem value="mdl3">
                      My optimised model - 3x5.pth
                    </MenuItem>
                  </Select>
                </FormControl>
                <FormControlLabel
                  style={{ marginTop:"20px", marginLeft: "0px", width: "100%", color: "#D9D9D9" }}
                  label="μManager integration"
                  labelPlacement="top"
                  control={
                    <Switch
                      checked={this.state.MM_integration}
                      onChange={this.MM_integration_handler.bind(this)}
                      value="useCloud"
                      color="primary"
                      size="small"
                    />
                  }
                />
              </ThemeProvider>
            </Grid>
          </Grid>
        </Grid>
      </>
    );
  }
}
