import { Route, Routes } from "react-router-dom";

import { NavRail } from "./app/NavRail";
import { TransportBar } from "./app/TransportBar";
import { Library } from "./screens/Library";
import { Placeholder } from "./screens/Placeholder";
import { RenderJobs } from "./screens/RenderJobs";

export default function App() {
  return (
    <div className="shell">
      <NavRail />
      <main className="main">
        <Routes>
          <Route path="/" element={<Library />} />
          <Route path="/listen" element={<Placeholder title="Listen" section="M6c-5" />} />
          <Route path="/review" element={<Placeholder title="Character Review" section="M6c-3" />} />
          <Route path="/voices" element={<Placeholder title="Voice Studio" section="M6c-4" />} />
          <Route path="/render" element={<RenderJobs />} />
        </Routes>
      </main>
      <TransportBar />
    </div>
  );
}
