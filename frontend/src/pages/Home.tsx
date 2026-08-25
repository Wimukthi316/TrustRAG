import type { CSSProperties } from "react";
import {
  ArrowRight,
  CircleHelp,
  FileText,
  Highlighter,
  MessageSquareText,
  Quote,
  ScanText,
  Scale,
  SearchCheck,
  ShieldAlert,
  ShieldCheck,
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

/**
 * The hero's floating glyphs.
 *
 * Decoration, and it says so: `aria-hidden`, `pointer-events-none`, and gone
 * below `xl`. Every icon is one the rest of the interface actually uses -- a
 * document, a mark, a scale, a shield -- so the field reads as this product
 * rather than as generic startup confetti.
 *
 * Every position is right of 60%. The headline caps at 42rem, so below a 1280px
 * viewport the two would start to collide and the whole field is dropped
 * instead of being allowed to drift across the text.
 */
const GLYPHS: {
  Icon: typeof FileText;
  left: string;
  top: string;
  size: number;
  tilt: number;
  delay: number;
  dur: number;
  opacity: number;
}[] = [
  { Icon: FileText, left: "62%", top: "14%", size: 56, tilt: -8, delay: 0, dur: 11, opacity: 0.95 },
  { Icon: Highlighter, left: "73%", top: "30%", size: 68, tilt: 7, delay: 1.4, dur: 13, opacity: 1 },
  { Icon: ShieldCheck, left: "89%", top: "12%", size: 46, tilt: 12, delay: 2.6, dur: 10, opacity: 0.8 },
  { Icon: ScanText, left: "66%", top: "60%", size: 60, tilt: 5, delay: 0.7, dur: 12, opacity: 0.9 },
  { Icon: Quote, left: "83%", top: "68%", size: 42, tilt: -11, delay: 3.4, dur: 14, opacity: 0.75 },
  { Icon: SearchCheck, left: "92%", top: "45%", size: 48, tilt: -5, delay: 2.0, dur: 11.5, opacity: 0.85 },
  { Icon: Scale, left: "76%", top: "85%", size: 40, tilt: 9, delay: 3.0, dur: 12.5, opacity: 0.7 },
  { Icon: MessageSquareText, left: "60%", top: "84%", size: 38, tilt: -14, delay: 4.1, dur: 13.5, opacity: 0.65 },
];

function GlyphField() {
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 hidden select-none xl:block"
    >
      {GLYPHS.map(({ Icon, left, top, size, tilt, delay, dur, opacity }, i) => (
        <span
          key={i}
          className="glyph glass-dark absolute flex items-center justify-center rounded-2xl text-white"
          style={
            {
              left,
              top,
              width: size,
              height: size,
              opacity,
              animationDelay: `${delay}s`,
              "--dur": `${dur}s`,
              "--tilt": `${tilt}deg`,
            } as CSSProperties
          }
        >
          <Icon style={{ width: size * 0.42, height: size * 0.42 }} />
        </span>
      ))}
    </div>
  );
}

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
      <section className="hero-wash relative flex min-h-[92svh] items-center overflow-hidden">
        <GlyphField />
        <div className="relative mx-auto w-full max-w-[86rem] px-5 pb-28 pt-36 sm:px-8 sm:pb-32 sm:pt-40 lg:px-12">
          <Reveal>
            <h1 className="t-display max-w-2xl text-white">
              AI writes things the document never said.
            </h1>
          </Reveal>

          <Reveal delay={0.06}>
            <p className="t-lead mt-6 max-w-xl text-brand-100/80">
              Give an AI a document and ask it a question, and most of the answer
              will be right. The trouble is the part that isn't — it reads exactly
              like the part that is. TrustRAG reads the answer against the
              document, marks the words the document does not back up, and tells
              you how sure it is in a number you can actually rely on.
            </p>
          </Reveal>

          <Reveal delay={0.12}>
            <div className="mt-10 flex flex-wrap gap-3">
              <Button onClick={onDemo}>
                See it on a real example
                <ArrowRight className="h-4 w-4" aria-hidden />
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
              <Card className="p-6 sm:p-8">
                <p className="t-eyebrow text-slate-500">What the AI wrote</p>
                <p className="t-answer mt-4 text-slate-800">
                  <span className={`mark ${VERDICT_SKIN.flag.mark}`}>
                    {specimen.span.text}
                  </span>
                </p>
                <div
                  className={`mt-6 inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-[0.8125rem] font-semibold ${VERDICT_SKIN.flag.chip}`}
                >
                  <ShieldAlert className="h-4 w-4" aria-hidden />
                  Not in the document
                </div>
              </Card>

              <Card className="bg-slate-50 p-6 sm:p-8">
                <p className="t-eyebrow text-slate-500">
                  {specimen.span.evidence_sentence
                    ? "What the document actually says"
                    : "What the document says about it"}
                </p>
                <p className="t-lead mt-4 text-slate-700">
                  {specimen.span.evidence_sentence ? (
                    <>
                      <Quote className="mr-2 inline h-4 w-4 -translate-y-0.5 text-slate-400" aria-hidden />
                      {specimen.span.evidence_sentence}
                    </>
                  ) : (
                    "Nothing. We read it one sentence at a time and not one of them backs this up."
                  )}
                </p>
                <p className="t-body mt-6 text-slate-500">
                  A person who had not read the article would never have caught
                  it. That is the whole difficulty — a made-up detail is written
                  in the same confident voice as a true one.
                </p>
              </Card>
            </div>
          </Reveal>
        ) : (
          <Reveal>
            <Card className="p-8">
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
                <Card className={`h-full p-6 ring-1 ring-inset ${skin.ring}`}>
                  <span
                    className={`inline-flex h-11 w-11 items-center justify-center rounded-xl ${skin.chip}`}
                  >
                    <Icon className="h-5 w-5" aria-hidden />
                  </span>
                  <h3 className="t-h3 mt-5 text-slate-900">{v.title}</h3>
                  <p className="t-body mt-3 text-slate-600">{v.body}</p>
                </Card>
              </Reveal>
            );
          })}
        </div>

        <Reveal delay={0.2}>
          <div className="mt-8 rounded-2xl border border-slate-200 bg-white px-6 py-5">
            <p className="t-small font-medium text-slate-500">
              This is the key to every colour you will see on the next page.
            </p>
            <VerdictLegend className="mt-4" />
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

        {/* A grid rather than a stack. Five stacked rows leave two thirds of a
            wide screen empty and push the next band a screen further down; the
            steps are short enough to read three across. */}
        <ol className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {STEPS.map((step, i) => (
            <Reveal key={step.title} delay={i * 0.05}>
              <li className="h-full rounded-2xl border border-slate-200 bg-white p-5">
                <span className="tnum flex h-8 w-8 items-center justify-center rounded-lg bg-brand-50 text-[0.875rem] font-semibold text-brand-700">
                  {i + 1}
                </span>
                <h3 className="t-h3 mt-4 text-slate-900">{step.title}</h3>
                <p className="t-body mt-2 text-slate-600">{step.plain}</p>
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
              <Card className="h-full p-6">
                <p className="tnum text-[2.5rem] font-semibold leading-none tracking-tight text-slate-900">
                  {item.value}
                  <span className="ml-1.5 text-lg font-medium text-slate-400">
                    {item.unit}
                  </span>
                </p>
                <h3 className="t-h3 mt-4 text-slate-900">{item.title}</h3>
                <p className="t-small mt-2.5 text-slate-600">{item.floor}</p>
              </Card>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.2}>
          <div className="mt-8">
            <Button variant="secondary" onClick={onProof}>
              <FileText className="h-4 w-4" aria-hidden />
              See the evidence behind these
            </Button>
          </div>
        </Reveal>
      </Band>

      {/* ---------------------------------------------------------------- */}
      <Band tone="dark">
        <div className="max-w-3xl">
          <h2 className="t-h2 text-white">Watch it read one.</h2>
          <p className="t-lead mt-4 text-brand-100/85">
            Pick one of three real answers, or paste your own document and see
            what comes back. The next page walks through it a step at a time.
          </p>
          <div className="mt-7">
            <Button variant="onDark" onClick={onDemo}>
              See it work
              <ArrowRight className="h-4 w-4" aria-hidden />
            </Button>
          </div>
        </div>
      </Band>
    </>
  );
}
