import React from "react";
import { createRoot } from "react-dom/client";
import PhotonSim from "./PhotonSim.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <PhotonSim />
  </React.StrictMode>
);
