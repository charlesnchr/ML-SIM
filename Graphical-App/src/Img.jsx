import React, { Component } from "react";
import CheckCircleIcon from "@material-ui/icons/CheckCircle";
import CheckCircleOutlinedIcon from "@material-ui/icons/CheckCircleOutlined";
import loadinggif from "./loading.gif";

const { ipcRenderer } = window.require("electron");
const log = window.require("electron-log");

var path = window.require("path");
var sess = require("./sess.js");


class Img extends Component {
    constructor(props) {
      log.info("CREATING new IMG component", props.filepath);
      super(props);
  
      let filepath = props.filepath;
      let dim = "N/A",
        imgsrc = null;
  
      this.state = {
        dim: dim,
        imgsrc: imgsrc,
        filepath: props.filepath,
        selectedBool: props.selectedBool,
        hoverBool: false,
      };
    }
  
    selectImage(filepath, e) {
      if (e.buttons == 1) {
        ipcRenderer.send('showImage',this.state.imgsrc)
      } else if (e.buttons == 2) {
        this.setState({ selectedBool: !this.state.selectedBool });
        sess.toggleSelectedFilepath(filepath);
      }
    }
  
    imageHoverOver(filepath, e) {
      if (e.buttons == 2) {
        this.setState({
          selectedBool: !this.state.selectedBool,
          hoverBool: true,
        });
        sess.toggleSelectedFilepath(filepath);
      } else {
        this.setState({ hoverBool: true });
      }
    }
  
    imageHoverOut(filepath, e) {
      this.setState({ hoverBool: false });
    }
  
    getThumb(filepath) {
      // if (
      //   !sess.thumbJobs.includes(filepath) &&
      //   !sess.thumbQueue.includes(filepath)
      // ) {
    //   log.info("inside gethumb", sess.thumbJobs.length, sess.thumbQueue.length);
      if (sess.thumbJobs.length > 1) {
        setTimeout(
          function() {
            this.getThumb(filepath);
          }.bind(this),
          100
        );
      } else {
        ipcRenderer.send("sendToPython", {
          cmd: "GetThumb",
          arg: filepath,
        });
        log.info('thumbJobs is:',sess.thumbJobs,sess.thumbJobs.length)
        sess.thumbJobs.push(filepath);
        log.info("starting a new job", filepath, sess.thumbJobs.length);
      }
      // }
    }
  
    componentDidMount() {
      if (path.extname(this.state.filepath) == ".png") {
        // showing results
        let dim = "2D image";
        let imgsrc = this.state.filepath;
        this.setState({ dim: dim, imgsrc: 'file://' + imgsrc });
      } else {  
        let thumb = sess.thumbdict[this.state.filepath];
  
        if (!thumb) {
          log.info("Thumb does not exist", this.state.filepath);
          this.getThumb(this.state.filepath);
        } else {
          this.setState({ dim: thumb.dim, imgsrc: thumb.src });
        }
      }
  
      ipcRenderer.on("thumb_" + this.state.filepath, (event, thumbpath, dim) => {
        // remove completed job
        let idx = sess.thumbJobs.indexOf(this.state.filepath);
        if (idx > -1) sess.thumbJobs.splice(idx, 1);
        
        // render if job was a success
        if (thumbpath !== "0") {
          thumbpath = 'file://' + thumbpath;
          sess.thumbdict[this.state.filepath] = { src: thumbpath, dim: dim };
          this.setState({ dim: dim, imgsrc: thumbpath });
        }
      });
    }


    render() {
      let fp = this.state.filepath;
      let style = {
        border: "1px solid black",
        boxSizing: "border-box",
        transition: "border-width 0.3s linear, border-color 1s linear",
      };
      let  check = [], divHover = [];
  
      if (this.state.selectedBool) {
        style = {
          border: "12px solid #e8f0fe",
          boxSizing: "border-box",
          transition: "border-width 0.13s linear",
        };
        check = [
          <CheckCircleOutlinedIcon
            key={fp + "-check"}
            style={{
              color: "white",
              position: "absolute",
              top: "3px",
              left: "3px",
              pointerEvents:'none',
            }}
          />,
          <CheckCircleIcon
            key={fp + "-checkbg"}
            style={{
              color: "#1A73E8",
              position: "absolute",
              top: "3px",
              left: "3px",
              pointerEvents:'none',
            }}
          />,
        ];
      } else if (this.state.hoverBool) {
        check = (
          <CheckCircleIcon
            key={fp + "-checkbg"}
            style={{
              color: "rgb(200,200,200)",
              position: "absolute",
              top: "3px",
              left: "3px",
              pointerEvents:'none',
            }}
          />
        );
        const styleHover = {
          background:
            "linear-gradient(to top,transparent 0%, rgba(0,0,0,0.7) 100%",
          width: sess.imgsize,
          height: 50,
          position: "absolute",
          margin: 0,
          pointerEvents:'none',
        };
        divHover = <div style={styleHover}></div>;
      } 
  
      const caption = (
        <span
          onClick={this.selectImage.bind(this, fp)}
          style={{
            position: "absolute",
            pointerEvents:'none',
            left: 0,
            bottom: 0,
            width: sess.imgsize,
            overflow: "hidden",
            whiteSpace: "nowrap",
            backgroundColor: "rgb(30,30,30,0.8)",
            color: "rgb(200,200,200)",
            paddingLeft: 5,
            paddingRight: 5,
            paddingTop: 2,
            paddingBottom: 2,
          }}
        >
          <span style={{ marginRight: 10, fontStyle: "italic" }}>
            {path.basename(fp)}
          </span>
          <br />
          Stack info: <span style={{ fontWeight: "bold" }}>{this.state.dim}</span>
        </span>
      );
  
      let img;
  
      if (this.state.imgsrc) {
        img = (
          <img
            alt="Collection"
            style={style}
            key={fp + "-img"}
            width={sess.imgsize}
            height={sess.imgsize}
            src={this.state.imgsrc}
          />
        );
      } else {
        img = (
          <div
            style={{
              ...style,
              ...{
                width: sess.imgsize,
                height: sess.imgsize,
                backgroundColor: "rgb(1,1,1,0.2)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: 0,
              },
            }}
            onClick={this.selectImage.bind(this, fp)}
          >
            <img src={loadinggif} width={50} height={50} />
          </div>
        );
      }
  
      return (
        <div
          key={fp + "-div"}
          onMouseDown={this.selectImage.bind(this, fp)}
          onMouseOver={this.imageHoverOver.bind(this, fp)}
          onMouseOut={this.imageHoverOut.bind(this, fp)}
          style={{ position: "relative" }}
        >
          {divHover}
          {img}
          {check}
          {caption}
        </div>
      );
    }
  }

  export default Img;