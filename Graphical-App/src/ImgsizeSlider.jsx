import React, { Component } from "react";

import Grid from "@material-ui/core/Grid";
import MenuItem from "@material-ui/core/MenuItem";
import Divider from "@material-ui/core/Divider";
import Select from "@material-ui/core/Select";
import FormControl from "@material-ui/core/FormControl";
import Slider from "@material-ui/core/Slider";
import ImageIcon from "@material-ui/icons/Image";


import { createMuiTheme, ThemeProvider } from "@material-ui/core/styles";
const log = window.require("electron-log");

const darktheme = createMuiTheme({
    palette: {
      type: "dark",
    },
    typography: {
      fontFamily: "Roboto",
      body1: {
        fontFamily: "Roboto",
        fontSize: 10,
      },
      body2: {
        fontFamily: "Roboto",
        fontSize: 10,
      },
      div: {
        fontFamily: "Roboto",
      },
      ul: {
        fontSize: 10,
      },
    },
  });

var sess = require("./sess.js");


export default class ImgsizeSlider extends Component {
    constructor(props) {
      super(props);
      this.state = {
        imgsize: props.imgsize,
        handler: props.handler,
        showSelect: true,
      };
      sess.showHideSelect = this.showHideSelect.bind(this);
    }
  
    imgsizeSlider(event, value) {
      this.setState({ imgsize: value });
      this.state.handler(this.state.imgsize);
    }
  
    handleSelect(e) {
      log.info('Sorting by', e.target.value);
      sess.resort = true;
      sess.sortBy = e.target.value;
      sess.readFolders(sess.displayedFolders, e.target.value);
    }
  
    showHideSelect(showSelect) {
      this.setState({ showSelect: showSelect });
    }
  
    render() {
      return (
        <Grid container direction="row" justify="flex-end" alignItems="center">
          <Grid item>
            <Grid container direction="row" justify="flex-end">
              <Grid item></Grid>
              <Grid item>
                <ThemeProvider theme={darktheme}>
                  <FormControl disabled={!this.state.showSelect}>
                    <Select
                      inputProps={{ className: "selectInput" }}
                      defaultValue="filename"
                      onChange={this.handleSelect.bind(this)}
                    >
                      <MenuItem style={{ fontSize: "14px" }} value="filename">
                        Filename
                      </MenuItem>
                      <MenuItem style={{ fontSize: "14px" }} value="dateCreated">
                        Date created
                      </MenuItem>
                      <MenuItem style={{ fontSize: "14px" }} value="folderOrder">
                        Top folder first
                      </MenuItem>
                      <MenuItem
                        style={{ fontSize: "14px" }}
                        value="reverseFolderOrder"
                      >
                        Bottom folder first
                      </MenuItem>
                    </Select>
                  </FormControl>
                </ThemeProvider>
              </Grid>
            </Grid>
          </Grid>
          <Divider
            orientation="vertical"
            style={{
              margin: "0 20px 0 10px",
              height: 25,
              background: "rgba(50,50,50)",
            }}
          />
          <Grid className="nondraggable" item>
            <ImageIcon style={{ fontSize: 18 }} />
            &nbsp;&nbsp;&nbsp;
          </Grid>
          <Grid className="nondraggable" item>
            <Slider
              value={this.state.imgsize}
              min={200}
              max={500}
              onChange={this.imgsizeSlider.bind(this)}
              step={50}
              style={{ width: "150px" }}
              aria-labelledby="continuous-slider"
              disabled={!this.state.showSelect}
            />
          </Grid>
          <Grid className="nondraggable" item>
            &nbsp;&nbsp;&nbsp;
            <ImageIcon />
          </Grid>
        </Grid>
      );
    }
  }