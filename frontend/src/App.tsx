import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import Nav, { type View } from "./components/Nav";
import Home from "./pages/Home";
import Demo from "./pages/Demo";
import Proof from "./pages/Proof";
import { getRecordedRuns, type RecordedCase } from "./recorded";

/**
 * Three pages, not three panels.
 *
 *   Home     what the thing is, for someone who has never heard of RAG
 *   Demo     what it does, on a document, one step at a time
 *   Proof    why anyone should believe the number it just produced
 *
 * They were one page with tabs before, and everything was on it at once. The
 * people this is shown to are not the people who built it; they get one idea
 * per screen, in the order the idea needs.
 */

export default function App() {
  const [view, setView] = useState<View>("home");

  // Recorded runs are loaded once and shared. They are **recorded, not
  // simulated**: `src/common/recorded_analysis.py` rebuilds each one from the
  // components' own artefacts, so every field is what the live endpoint would
  // have returned. Having them means the demo still works when the backend
  // does not, which in front of a room is the only thing that matters.
  const [recorded, setRecorded] = useState<RecordedCase[]>([]);

  useEffect(() => {
    getRecordedRuns()
      .then((runs) => setRecorded(runs.cases))
      .catch(() => setRecorded([]));
  }, []);

  // Moving between pages should start at the top of the new one.
  //
  // Block body, not a concise arrow. An effect returns its cleanup function, so
  // `() => window.scrollTo(...)` hands React whatever scrollTo returned and
  // React then tries to call it when the effect re-runs. That is the
  // "destroy is not a function" crash, and it takes the whole tree down.
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "auto" });
  }, [view]);

  return (
    <div className="min-h-full">
      <Nav view={view} onView={setView} overDark={view === "home"} />

      <main>
        <AnimatePresence mode="wait">
          <motion.div
            key={view}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
          >
            {view === "home" && (
              <Home
                recorded={recorded}
                onDemo={() => setView("demo")}
                onProof={() => setView("proof")}
              />
            )}
            {view === "demo" && (
              <Demo recorded={recorded} onProof={() => setView("proof")} />
            )}
            {view === "proof" && <Proof onDemo={() => setView("demo")} />}
          </motion.div>
        </AnimatePresence>
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-[86rem] px-5 py-8 sm:px-8 lg:px-12">
          <p className="t-small max-w-3xl text-slate-500">
            TrustRAG — span-level calibrated hallucination detection for
            retrieval-augmented generation. Research prototype, SLIIT IT4010.
            Every number shown here came from a run on data held back from
            training; where a result is negative it is reported at the same size
            as the positive ones.
          </p>
        </div>
      </footer>
    </div>
  );
}
