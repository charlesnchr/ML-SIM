import React, { Component } from "react";
import Grid from "@material-ui/core/Grid";
import LinearProgress from "@material-ui/core/LinearProgress";


const { ipcRenderer } = window.require("electron");
const log = window.require("electron-log");
var sess = require("./sess.js");


class CalcStatus extends Component {
    constructor(props) {
      super(props);
      this.state = { status: null, progress: null };
    }
  
    isCalcFeaturesFinished() {
      if (this.state.status == null && this.state.progress == null) return true;
      else return false;
    }
  
    componentDidMount() {
      sess.isCalcFeaturesFinished = this.isCalcFeaturesFinished.bind(this);
  
      ipcRenderer.on("status", (event, cmd, msg) => {
        let p, s;
  
        if (cmd === "i") {
          // indexing

          msg = msg.split(",");
          var taskName = msg[0];
          var n1 = Number.parseInt(msg[1]);
          var n2 = Number.parseInt(msg[2]);
          p =
            Number.parseInt(
              (100.0 * Number.parseFloat(n1)) / Number.parseFloat(n2)
            ) + 1;
          s = (
            <Grid container>
              <Grid style={{ textAlign: "left" }} item xs>
                {taskName}
              </Grid>
              <Grid style={{marginRight:15}} item xs>
                {n1 + "/" + n2}
              </Grid>
            </Grid>
          );
          if (n1 % 10 === 0) log.info("indexing status", n1, "/", n2);
        } else if (cmd === "l") {
          // loading
          p = Number.parseInt(msg);
          s = (
            <Grid container>
              <Grid style={{ textAlign: "left" }} item xs>
                {"Loading:"}
              </Grid>
              <Grid item xs>
                {p + " %"}
              </Grid>
            </Grid>
          );
          if (p % 50 === 0) log.info("loading status", p);
        } else if (cmd === "h") {
          // hashing
          p = Number.parseInt(msg);
          s = (
            <Grid container>
              <Grid style={{ textAlign: "left" }} item xs>
                {"Scanning:"}
              </Grid>
              <Grid item xs>
                {p + " %"}
              </Grid>
            </Grid>
          );
          if (p % 50 === 0) log.info("scanning status", p);
        } else if (cmd === "c") {
          // computing search tree
          s = (
            <Grid container>
              <Grid style={{ textAlign: "left" }} item xs>
                {"Computing search tree"}
              </Grid>
            </Grid>
          );
          p = -1;
        } else if (cmd === "m") {
          // loading model
          s = (
            <Grid container>
              <Grid style={{ textAlign: "left" }} item xs>
                {"Loading model"}
              </Grid>
            </Grid>
          );
          p = -1;
        } else if (cmd === "d") {
          s = null;
          p = null;
        }
  
        this.setState({ status: s, progress: p });
      });
    }
  
    render() {
      var progressBar = this.state.progress ? (
        <LinearProgress style={{marginRight:15}} variant={"indeterminate"} value={this.state.progress} />
      ) : (
        ""
      );
      return (
        <span style={{paddingRight:20}}>
          {progressBar}
          {this.state.status}
        </span>
      );
    }
  }

  export default CalcStatus;