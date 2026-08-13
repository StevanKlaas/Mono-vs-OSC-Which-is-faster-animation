import React from "react";
import { createRoot } from "react-dom/client";
import { Analytics } from "@vercel/analytics/react";
import PhotonSim from "./PhotonSim";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <PhotonSim />
    <Analytics />
  </React.StrictMode>
);
