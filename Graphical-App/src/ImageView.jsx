import React, { Component } from "react";
import Grid from "@material-ui/core/Grid";

var sess = require("./sess.js");

const { ipcRenderer } = window.require("electron");
const log = window.require("electron-log");

class ImageView extends Component {
  constructor(props) {
    super(props);

    this.state = { filepath: "N/A", appWidth: window.innerWidth };
  }

  componentDidMount() {
    log.info("setting event handler filepathArugment");
    ipcRenderer.on("filepathArgument", (event, filepath) => {
      log.info("received filepathargument", filepath);
      this.setState({ filepath: filepath });
      document.title = filepath;
    });

    window.addEventListener("resize", this.setWidth.bind(this));
  }

  setWidth() {
    if (this.state.appWidth !== window.innerWidth) {
      this.setState({ appWidth: window.innerWidth });
    }
  }

  render() {
    // if (this.state.filepath !== "N/A") {
    //   img = <img src={this.state.filepath} />;
    // }
    let header = (
      <div
        className="App-header draggable"
        style={{
          position: "fixed",
          width: this.state.appWidth,
          zIndex: 100,
          height: sess.headerHeight
        }}
      >
        <Grid container alignItems="center" justify="space-between" spacing={2} style={{paddingTop:30}}>
        {this.state.filepath}  
        </Grid>
      </div>
    );
    let imgdiv = (
      <div
        style={{
          display: "table",
          marginLeft: "auto",
          width: 800,
          height: 600,
          marginRight: "auto",
          paddingTop: sess.headerHeight,
          backgroundImage: `url("${this.state.filepath}")`,
          backgroundRepeat: "no-repeat",
          backgroundPosition: "center",
          backgroundSize: "cover",
        }}
      >
        
      </div>
    );
    return <>{header}{imgdiv}</>;
  }
}
export default ImageView;
