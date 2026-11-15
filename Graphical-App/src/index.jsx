import React from "react";
import ReactDOM from "react-dom";
import App from "./App";
import ImageView from "./ImageView";
import "./index.css";
import "typeface-roboto";

import { HashRouter as Router, Switch, Route } from "react-router-dom";

let router = (
  <Router>
    <Switch>
      <Route exact path="/" component={App}/>
      <Route exact path="/imageview" component={ImageView}/>
    </Switch>
  </Router>
);
ReactDOM.render(router, document.getElementById("root"));

// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// No Node.js APIs are available in this process because
// `nodeIntegration` is turned off. Use `preload.js` to
// selectively enable features needed in the rendering
// process.
