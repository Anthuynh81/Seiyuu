import { Route, Routes } from "react-router-dom";

import { NavRail } from "./app/NavRail";
import { TransportBar } from "./app/TransportBar";
import { Library } from "./screens/Library";
import { Placeholder } from "./screens/Placeholder";
import { RenderJobs } from "./screens/RenderJobs";
import { Review } from "./screens/Review";
import { Voices } from "./screens/Voices";

export default function App() {
  return (
    <div className="shell">
      <NavRail />
      <main className="main">
        <Routes>
          <Route path="/" element={<Library />} />
          <Route path="/listen" element={<Placeholder title="Listen" section="M6c-5" />} />
          <Route path="/review" element={<Review />} />
          <Route path="/voices" element={<Voices />} />
          <Route path="/render" element={<RenderJobs />} />
        </Routes>
      </main>
      <TransportBar />
    </div>
  );
}
