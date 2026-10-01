import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import Nav, { type View } from "./components/Nav";
import Home from "./pages/Home";
import Demo from "./pages/Demo";
import Proof from "./pages/Proof";
import Adaptive from "./pages/Adaptive";
import History from "./pages/History";
import { getRecordedRuns, type RecordedCase } from "./recorded";
import type { AnalysisResult } from "./types";

/**
 * Three pages, not three panels.
 *
 *   Home     what the thing is, for someone who has never heard of RAG
 *   Demo     what it does, on a document, one step at a time
 *   Proof    why anyone should believe the number it just produced
 *   History  everything this person has already checked
 *
 * They were one page with tabs before, and everything was on it at once. The
 * people this is shown to are not the people who built it; they get one idea
 * per screen, in the order the idea needs.
 *
 * There is no footer. The one that was here restated the project's name and
 * three claims already made better further up, which is what a footer usually
 * is. Each page now ends on the dark band that tells you where to go next.
 */

export default function App() {
  const [view, setView] = useState<View>("home");

  // Recorded runs are loaded once and shared. They are **recorded, not
  // simulated**: `src/common/recorded_analysis.py` rebuilds each one from the
  // components' own artefacts, so every field is what the live endpoint would
  // have returned. Having them means the demo still works when the backend
  // does not, which in front of a room is the only thing that matters.
  const [recorded, setRecorded] = useState<RecordedCase[]>([]);

  // The settings the dial may offer. They come out of the same file as the
  // cases, so the dial can never offer a position that was not built.
  const [alphas, setAlphas] = useState<number[]>([]);

  // A check opened from the history page. It is shown on the demo page rather
  // than on a fourth kind of screen, because a saved result and a fresh one are
  // the same thing and deserve the same six bands.
  const [opened, setOpened] = useState<AnalysisResult | null>(null);

  useEffect(() => {
    getRecordedRuns()
      .then((runs) => {
        setRecorded(runs.cases);
        setAlphas((runs.alphas ?? [String(runs.alpha)]).map(Number));
      })
      .catch(() => {
        setRecorded([]);
        setAlphas([]);
      });
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

      {/* The nav is fixed and floats clear of the top edge, so it takes no room
          in the flow. The home page opens on a hero that is meant to run under
          it; every other page has to start below it instead. */}
      <main className={view === "home" ? undefined : "pt-24"}>
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
              <Demo
                recorded={recorded}
                alphas={alphas}
                opened={opened}
                onProof={() => setView("proof")}
              />
            )}
            {view === "proof" && <Proof onDemo={() => setView("demo")} />}
            {view === "adaptive" && <Adaptive onProof={() => setView("proof")} />}
            {view === "history" && (
              <History
                onOpen={(result) => {
                  setOpened(result);
                  setView("demo");
                }}
                onDemo={() => {
                  setOpened(null);
                  setView("demo");
                }}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
