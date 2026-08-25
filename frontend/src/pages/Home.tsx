import {
  ArrowRight,
  CircleHelp,
  FileText,
  Quote,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { SCORECARD, STEPS, VERDICT_CARDS } from "../copy";
import type { RecordedCase } from "../recorded";
import type { Span } from "../types";
import { VerdictLegend } from "../components/Marked";
import { Band, BandHead, Button, Card, Reveal } from "../components/ui";

/**
 * The page the site opens on.
 *
 * It has one job: a person who has never heard of retrieval-augmented
 * generation should reach the bottom of it knowing what the thing does and why
 * anyone would want it. So it is a stack of bands with one idea each, told in
 * the order the idea needs -- the problem, then the three answers, then how, then
 * what we can prove -- rather than a dashboard of panels competing for a glance.
 *
 * The specimen in the second band is not written copy. It is pulled out of a
 * recorded run, so the thing on the landing page is the thing the system
 * actually produced.
 */

const VERDICT_ICON = {
  flag: ShieldAlert,
  abstain: CircleHelp,
  pass: ShieldCheck,
} as const;

const VERDICT_SKIN = {
  flag: {
    ring: "ring-flag-ring",
    chip: "bg-flag-soft text-flag-ink",
    mark: "bg-flag-soft text-flag-ink [--mark-ring:var(--color-flag-ring)]",
  },
  abstain: {
    ring: "ring-abstain-ring",
    chip: "bg-abstain-soft text-abstain-ink",
    mark: "bg-abstain-soft text-abstain-ink [--mark-ring:var(--color-abstain-ring)]",
  },
  pass: {
    ring: "ring-pass-ring",
    chip: "bg-pass-soft text-pass-ink",
    mark: "bg-pass-soft text-pass-ink [--mark-ring:var(--color-pass-ring)]",
  },
} as const;

/** The shortest honest specimen available: one flagged passage and its source. */
function pickSpecimen(
  recorded: RecordedCase[],
): { span: Span; caseTitle: string } | null {
  for (const preferred of ["news-summary", "carbon-footprint"]) {
    const item = recorded.find((c) => c.slug === preferred);
    if (!item) continue;
    const span = item.result.spans.find(
      (s) => s.conformal_decision === "flag" && s.text.length < 90,
    );
    if (span) return { span, caseTitle: item.title };
  }
  return null;
}

export default function Home({
  recorded,
  onDemo,
  onProof,
}: {
  recorded: RecordedCase[];
  onDemo: () => void;
  onProof: () => void;
}) {
  const specimen = pickSpecimen(recorded);

  return (
    <>
      {/* ---------------------------------------------------------------- */}
      <section className="hero-wash relative -mt-[73px] overflow-hidden pt-[73px]">
        <div className="mx-auto max-w-6xl px-6 py-24 sm:px-10 sm:py-32">
          <Reveal>
            <span className="inline-flex items-center gap-2.5 rounded-full bg-white/10 px-4 py-2 text-[0.875rem] font-medium text-brand-100 ring-1 ring-inset ring-white/20">
              <Sparkles className="h-4 w-4" aria-hidden />
              Final-year research project · SLIIT
            </span>
          </Reveal>

          <Reveal delay={0.05}>
            <h1 className="t-display mt-8 max-w-4xl text-white">
              AI writes things the document never said.
            </h1>
          </Reveal>

          <Reveal delay={0.1}>
            <p className="t-lead mt-8 max-w-2xl text-brand-100/85">
              Give an AI a document and ask it a question, and most of the answer
              will be right. The trouble is the part that isn't — it reads exactly
              like the part that is. TrustRAG reads the answer against the
              document, marks the words the document does not back up, and tells
              you how sure it is in a number you can actually rely on.
            </p>
          </Reveal>

          <Reveal delay={0.15}>
            <div className="mt-11 flex flex-wrap gap-4">
              <Button onClick={onDemo}>
                See it on a real example
                <ArrowRight className="h-5 w-5" aria-hidden />
              </Button>
              <Button variant="onDark" onClick={onProof}>
                Why believe it
              </Button>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ---------------------------------------------------------------- */}
      <Band tone="white">
        <BandHead
          eyebrow="The problem"
          title="It looks exactly like the truth."
          lead={
            specimen
              ? "This came out of a real run, not out of our imagination. An AI was given a news article and asked to summarise it. Here is one line it produced, next to the line in the article it was supposed to come from. Read them side by side."
              : "An AI is given a document and asked to summarise it. Most of what comes back is fine. One clause is not."
          }
        />

        {specimen ? (
          <Reveal>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card className="p-8 sm:p-10">
                <p className="t-eyebrow text-slate-500">What the AI wrote</p>
                <p className="t-answer mt-6 text-slate-800">
                  <span className={`mark ${VERDICT_SKIN.flag.mark}`}>
                    {specimen.span.text}
                  </span>
                </p>
                <div
                  className={`mt-8 inline-flex items-center gap-2.5 rounded-full px-4 py-2 text-[0.9375rem] font-semibold ${VERDICT_SKIN.flag.chip}`}
                >
                  <ShieldAlert className="h-4.5 w-4.5" aria-hidden />
                  Not in the document
                </div>
              </Card>

              <Card className="bg-slate-50 p-8 sm:p-10">
                <p className="t-eyebrow text-slate-500">
                  {specimen.span.evidence_sentence
                    ? "What the document actually says"
                    : "What the document says about it"}
                </p>
                <p className="t-lead mt-6 text-slate-700">
                  {specimen.span.evidence_sentence ? (
                    <>
                      <Quote className="mr-2 inline h-5 w-5 -translate-y-1 text-slate-400" aria-hidden />
                      {specimen.span.evidence_sentence}
                    </>
                  ) : (
                    "Nothing. We read it one sentence at a time and not one of them backs this up."
                  )}
                </p>
                <p className="t-body mt-8 text-slate-500">
                  A person who had not read the article would never have caught
                  it. That is the whole difficulty — a made-up detail is written
                  in the same confident voice as a true one.
                </p>
              </Card>
            </div>
          </Reveal>
        ) : (
          <Reveal>
            <Card className="p-10">
              <p className="t-lead text-slate-600">
                The recorded examples are still loading. Open “See it work” for
                the full version.
              </p>
            </Card>
          </Reveal>
        )}
      </Band>

      {/* ---------------------------------------------------------------- */}
      <Band tone="plain">
        <BandHead
          eyebrow="What it can say"
          title="Three answers, never two."
          lead="Almost every tool of this kind gives you a yes or a no, and quietly guesses when it does not know. This one has a third answer, and it uses it about half the time."
        />

        <div className="grid gap-6 md:grid-cols-3">
          {VERDICT_CARDS.map((v, i) => {
            const Icon = VERDICT_ICON[v.tone];
            const skin = VERDICT_SKIN[v.tone];
            return (
              <Reveal key={v.tone} delay={i * 0.07}>
                <Card className={`h-full p-8 ring-1 ring-inset ${skin.ring}`}>
                  <span
                    className={`inline-flex h-14 w-14 items-center justify-center rounded-2xl ${skin.chip}`}
                  >
                    <Icon className="h-7 w-7" aria-hidden />
                  </span>
                  <h3 className="t-h3 mt-7 text-slate-900">{v.title}</h3>
                  <p className="t-body mt-4 text-slate-600">{v.body}</p>
                </Card>
              </Reveal>
            );
          })}
        </div>

        <Reveal delay={0.2}>
          <div className="mt-12 rounded-2xl border border-slate-200 bg-white px-8 py-7">
            <p className="t-small font-medium text-slate-500">
              This is the key to every colour you will see on the next page.
            </p>
            <VerdictLegend className="mt-5" />
          </div>
        </Reveal>
      </Band>

      {/* ---------------------------------------------------------------- */}
      <Band tone="white">
        <BandHead
          eyebrow="How it works"
          title="Five steps, and none of them are magic."
          lead="Nothing here needs the internet, another AI, or a person who knows what they are looking for. It only ever compares the answer to the document it was given."
        />

        <ol className="space-y-2">
          {STEPS.map((step, i) => (
            <Reveal key={step.title} delay={i * 0.05}>
              <li className="flex gap-7 rounded-2xl px-3 py-7 transition hover:bg-slate-50 sm:px-6">
                <span className="tnum flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-brand-50 text-2xl font-semibold text-brand-700">
                  {i + 1}
                </span>
                <div className="min-w-0 pt-1.5">
                  <h3 className="t-h3 text-slate-900">{step.title}</h3>
                  <p className="t-body mt-3 max-w-2xl text-slate-600">{step.plain}</p>
                </div>
              </li>
            </Reveal>
          ))}
        </ol>
      </Band>

      {/* ---------------------------------------------------------------- */}
      <Band tone="plain">
        <BandHead
          eyebrow="What we can show"
          title="Every number here came from a run we can repeat."
          lead="And every one of them is printed next to the thing it has to beat, because a score with its floor taken away is exactly the trick this project was built to complain about."
        />

        <div className="grid gap-6 md:grid-cols-3">
          {SCORECARD.map((item, i) => (
            <Reveal key={item.title} delay={i * 0.07}>
              <Card className="h-full p-8">
                <p className="tnum text-6xl font-semibold tracking-tight text-slate-900">
                  {item.value}
                  <span className="ml-2 text-2xl font-medium text-slate-400">
                    {item.unit}
                  </span>
                </p>
                <h3 className="t-h3 mt-6 text-slate-900">{item.title}</h3>
                <p className="t-small mt-4 text-slate-600">{item.floor}</p>
              </Card>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.2}>
          <div className="mt-12">
            <Button variant="secondary" onClick={onProof}>
              <FileText className="h-5 w-5" aria-hidden />
              See the evidence behind these
            </Button>
          </div>
        </Reveal>
      </Band>

      {/* ---------------------------------------------------------------- */}
      <Band tone="dark">
        <div className="max-w-3xl">
          <h2 className="t-h2 text-white">Watch it read one.</h2>
          <p className="t-lead mt-6 text-brand-100/85">
            Pick one of three real answers, or paste your own document and see
            what comes back. The next page walks through it a step at a time.
          </p>
          <div className="mt-10">
            <Button variant="onDark" onClick={onDemo}>
              See it work
              <ArrowRight className="h-5 w-5" aria-hidden />
            </Button>
          </div>
        </div>
      </Band>
    </>
  );
}
