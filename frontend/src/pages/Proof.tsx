import { useEffect, useState } from "react";
import { ArrowRight, Loader2, TriangleAlert } from "lucide-react";
import { FAULTS } from "../copy";
import { getProofCharts, type ProofCharts } from "../proof";
import {
  CoverageFigure,
  EvidenceFigure,
  ReliabilityFigure,
} from "../components/charts";
import { Band, BandHead, Button, Card, Reveal } from "../components/ui";

/**
 * Why anyone should believe the number on the previous page.
 *
 * Three pictures and nothing else. There were nine before, and the honest
 * assessment was that seven of them answered questions only the people who
 * built this were asking. The rest of the measurements are in the paper.
 *
 * Every figure is drawn from `public/proof-charts.json`, which the Python
 * writes out of the recorded runs. Nothing on this page is typed in by hand.
 *
 * Unlike the other two pages this one runs in a single narrow column. The wide
 * layout left each heading stranded in the top-left corner of a chart card four
 * times its width, which reads as an accident rather than as a choice. A column
 * the width of its own figures puts the question, the picture and the answer on
 * one axis, which is the order they are meant to be read in.
 */

/** The column every band on this page shares. */
function Column({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-4xl">{children}</div>;
}

export default function Proof({ onDemo }: { onDemo: () => void }) {
  const [data, setData] = useState<ProofCharts | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getProofCharts()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) {
    return (
      <Band tone="white">
        <BandHead
          title="The measurements did not load."
          lead={`Nothing was found to draw from: ${error}`}
        />
      </Band>
    );
  }

  if (!data) {
    return (
      <Band tone="white">
        <div className="flex items-center gap-4 text-slate-500">
          <Loader2 className="h-5 w-5 animate-spin" aria-hidden />
          <p className="t-lead">Loading the measurements…</p>
        </div>
      </Band>
    );
  }

  const { reliability, coverage, evidence } = data;
  const rawTop = reliability.before[reliability.before.length - 1];
  const fixedTop = reliability.after[reliability.after.length - 1];
  const heldAll = coverage.every((c) => c.held);
  const skipped = evidence.pool - evidence.returned;

  return (
    <>
      {/* ---------------------------------------------------------------- */}
      <Band tone="white">
        <div className="mx-auto max-w-2xl text-center">
          <p className="t-eyebrow text-brand-600">The evidence</p>
          <h1 className="t-title mt-4 text-slate-900">
            Three questions worth asking.
          </h1>
          <p className="t-lead mt-4 text-slate-600">
            Any system can print a percentage. These three pictures are the
            answers to the only questions that decide whether the percentage was
            worth printing — and each one was measured on data the system never
            saw while it was being built.
          </p>
        </div>
      </Band>

      {/* ---- 1 ---------------------------------------------------------- */}
      <Band tone="plain">
        <Column>
          <BandHead
            step={1}
            eyebrow="Question one"
            title="When it says 80%, is it right 80% of the time?"
            lead="Every dot is a group of passages the system gave roughly the same score to. Across the bottom is what it said the chance was. Up the side is what actually turned out to be true. If a system is honest, its dots sit on the dashed line."
          />

          <Reveal>
            <Card className="p-5 sm:p-8">
              <ReliabilityFigure data={reliability} />
            </Card>
          </Reveal>

          <Reveal delay={0.1}>
            <Callout title="What this shows">
              Straight out of the model, it was badly over-confident: when it said{" "}
              <Num>{Math.round(rawTop.confidence * 100)}%</Num> the real rate was{" "}
              <Num>{Math.round(rawTop.actual * 100)}%</Num>. After correction, when
              it says <Num>{Math.round(fixedTop.confidence * 100)}%</Num> the real
              rate is <Num>{Math.round(fixedTop.actual * 100)}%</Num>. Averaged over
              the whole range it went from being{" "}
              <Num>{Math.round(reliability.ece_before * 100)} points</Num> off to{" "}
              <Num>{Math.round(reliability.ece_after * 100)} points</Num> off —
              about{" "}
              <Num>
                {(reliability.ece_before / reliability.ece_after).toFixed(1)}×
              </Num>{" "}
              closer to the truth.
            </Callout>
          </Reveal>
        </Column>
      </Band>

      {/* ---- 2 ---------------------------------------------------------- */}
      <Band tone="white">
        <Column>
          <BandHead
            step={2}
            eyebrow="Question two"
            title="It makes a promise up front. Does it keep it?"
            lead="Before it reads anything you tell it how thorough to be — catch 9 in 10, catch 8 in 10, and so on. That is a promise made before the text is seen. Each row here is one of those promises and what the system then actually did."
          />

          <Reveal>
            <Card className="p-5 sm:p-8">
              <CoverageFigure data={coverage} />
            </Card>
          </Reveal>

          <Reveal delay={0.1}>
            <Callout title="What this shows">
              {heldAll ? (
                <>
                  It was asked at <Num>{coverage.length}</Num> different levels and
                  kept the promise at every one. Each dot lands inside the shaded
                  allowance — the small amount a single fair run is expected to fall
                  short by, purely from the luck of which examples ended up in the
                  test. That allowance is not a fudge factor: it is worked out from
                  how many examples there were, before the result was looked at.
                </>
              ) : (
                <>
                  It was asked at <Num>{coverage.length}</Num> different levels and
                  did not keep every one. The ones that fell outside the shaded
                  allowance are the misses, and they are shown at the same size as
                  the rest.
                </>
              )}
            </Callout>
          </Reveal>
        </Column>
      </Band>

      {/* ---- 3 ---------------------------------------------------------- */}
      <Band tone="plain">
        <Column>
          <BandHead
            step={3}
            eyebrow="Question three"
            title="Fine — but does it actually save anyone time?"
            lead="A tool that marks a sentence and then makes you re-read the whole document has not helped. Each square below is one sentence of a typical document. The coloured ones are the sentences the system hands you when it marks something."
          />

          <Reveal>
            <Card className="p-5 sm:p-8">
              <EvidenceFigure data={evidence} />
            </Card>
          </Reveal>

          <Reveal delay={0.1}>
            <Callout title="What this shows">
              A document in this test had about <Num>{evidence.pool.toFixed(1)}</Num>{" "}
              sentences in it. To settle a mark yourself you read about{" "}
              <Num>{evidence.returned.toFixed(1)}</Num> of them — the other{" "}
              <Num>{skipped.toFixed(1)}</Num> can be skipped. And this shortlist is
              not a guess either: it was built to contain the right sentence{" "}
              <Num>9 times in 10</Num>, and on the held-out data it contained it{" "}
              <Num>{(evidence.coverage * 100).toFixed(1)}%</Num> of the time.
            </Callout>
          </Reveal>
        </Column>
      </Band>

      {/* ---------------------------------------------------------------- */}
      <Band tone="white">
        <BandHead
          eyebrow="The other half"
          title="Where it falls down."
          lead="These are ours to report, and none of them were found by a reviewer. A results page that only contains wins is not a results page."
        />

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {FAULTS.map((fault, i) => (
            <Reveal key={fault.title} delay={Math.min(i, 4) * 0.05}>
              <Card className="h-full p-6">
                <span className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-abstain-soft text-abstain-ink">
                  <TriangleAlert className="h-4 w-4" aria-hidden />
                </span>
                <h3 className="t-h3 mt-4 text-slate-900">{fault.title}</h3>
                <p className="t-body mt-2.5 text-slate-600">{fault.body}</p>
              </Card>
            </Reveal>
          ))}
        </div>
      </Band>

      {/* ---------------------------------------------------------------- */}
      <Band tone="dark">
        <div className="max-w-3xl">
          <h2 className="t-h2 text-white">Now go and try to break it.</h2>
          <p className="t-lead mt-4 text-brand-100/85">
            Paste in a document of your own and see whether it marks the right
            things. If it marks the wrong ones, that is worth knowing too.
          </p>
          <div className="mt-7">
            <Button variant="onDark" onClick={onDemo}>
              Back to the examples
              <ArrowRight className="h-4 w-4" aria-hidden />
            </Button>
          </div>
        </div>
      </Band>
    </>
  );
}

function Callout({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-6 rounded-2xl border-l-4 border-brand-500 bg-white px-6 py-6 shadow-[0_1px_2px_rgba(15,23,42,0.04)] sm:px-8">
      <p className="t-eyebrow text-brand-600">{title}</p>
      <p className="t-lead mt-3 max-w-3xl text-slate-700">{children}</p>
    </div>
  );
}

function Num({ children }: { children: React.ReactNode }) {
  return <span className="tnum font-semibold text-slate-900">{children}</span>;
}
