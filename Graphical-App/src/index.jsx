import React from "react";
import ReactDOM from "react-dom";
import App from "./App";
import ImageView from "./ImageView";
import "./index.css";
import "@fontsource/roboto/300.css";
import "@fontsource/roboto/400.css";
import "@fontsource/roboto/500.css";
import "@fontsource/roboto/700.css";

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

