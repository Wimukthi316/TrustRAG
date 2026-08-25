import type { CSSProperties } from "react";
import {
  ArrowRight,
  CircleHelp,
  FileText,
  Highlighter,
  ListChecks,
  Percent,
  Quote,
  Scale,
  ScanText,
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
 * A window of the answer around one flagged passage.
 *
 * The hero shows the product doing its job rather than a field of icons, which
 * means it needs a piece of a real answer short enough to read at a glance. This
 * takes the text either side of a span and snaps the cut to a word boundary, so
 * the excerpt never opens or closes mid-word.
 */
function excerpt(
  answer: string,
  span: Span,
  before = 90,
  after = 90,
): { lead: string; marked: string; tail: string } {
  let from = Math.max(0, span.start - before);
  let to = Math.min(answer.length, span.end + after);
  if (from > 0) {
    const space = answer.indexOf(" ", from);
    from = space === -1 || space > span.start ? from : space + 1;
  }
  if (to < answer.length) {
    const space = answer.lastIndexOf(" ", to);
    to = space === -1 || space < span.end ? to : space;
  }
  return {
    lead: (from > 0 ? "… " : "") + answer.slice(from, span.start),
    marked: answer.slice(span.start, span.end),
    tail: answer.slice(span.end, to) + (to < answer.length ? " …" : ""),
  };
}

/** The shortest honest specimen available: one flagged passage and its source. */
function pickSpecimen(
  recorded: RecordedCase[],
): { span: Span; result: RecordedCase["result"]; caseTitle: string } | null {
  for (const preferred of ["news-summary", "carbon-footprint"]) {
    const item = recorded.find((c) => c.slug === preferred);
    if (!item) continue;
    // The shortest flagged passage that has a source sentence behind it.
    //
    // Shortest is doing real work rather than being arbitrary. A long mark wraps
    // across two lines of the card and renders as two separate boxes, and the
    // longest mark in a summary is usually its headline -- which reads as a
    // title rather than as a sentence, and makes a poor demonstration of a tool
    // whose whole point is marking words inside prose.
    const candidates = item.result.spans
      .filter((s) => s.conformal_decision === "flag" && s.text.length < 90)
      .sort((a, b) => a.text.length - b.text.length);
    const best = candidates.find((s) => s.evidence_sentence !== null) ?? candidates[0];
    if (best) return { span: best, result: item.result, caseTitle: item.title };
  }
  return null;
}

/**
 * The stat strip's labels.
 *
 * SCORECARD's titles are written for a card with room under them; the hero has
 * three columns and about four words each. Keyed off the title rather than the
 * index so reordering the scorecard cannot silently mislabel the hero.
 */
/**
 * A flagged passage the document says nothing at all about.
 *
 * The hero already shows a mark next to the sentence it was checked against.
 * The problem band needs the other half of the story, and it is the harder one:
 * an invented claim has nothing to point at. Falls back to whatever flagged
 * passage exists rather than showing an empty band.
 */
function pickInvented(recorded: RecordedCase[]) {
  for (const item of recorded) {
    // Longest, within reason. A three-word fragment is a weak thing to point at
    // and say "the document never mentions this"; the point lands when there is
    // enough of a claim on screen for a reader to form an opinion about it.
    const candidates = item.result.spans
      .filter((s) => s.conformal_decision === "flag" && s.evidence_sentence === null)
      .filter((s) => s.text.length < 120)
      .sort((a, b) => b.text.length - a.text.length);
    if (candidates[0]) {
      return { span: candidates[0], result: item.result, caseTitle: item.title };
    }
  }
  return null;
}

const HERO_STAT_LABEL: Record<string, string> = {
  "How often it gets a whole answer right": "whole answers right",
  "When it makes a promise, it keeps it": "promises kept",
  "Its percentages mean what they say": "closer to honest",
};

/**
 * The hero's visual.
 *
 * Small tiles, arranged rather than scattered.
 *
 * This was a drifting field of icons once and it read as confetti, for two
 * reasons worth writing down. The positions were random, so the eye found no
 * order in them; and the only thing holding it together was an animation, which
 * `prefers-reduced-motion` stops dead -- and that setting is on by default on
 * any machine with Windows animation effects switched off, which is most
 * laptops. What most people actually saw was eight static shapes at random.
 *
 * So: fixed positions on a deliberate diagonal, hairlines drawn between the
 * ones that belong together, three sizes to give it depth, and one accent tile
 * at the optical centre. It has to be finished standing perfectly still, and the
 * drift is a bonus for the people whose machines allow it.
 *
 * Every icon is one the interface actually uses. A document, a mark, a shield,
 * a quotation, a pair of scales -- the product's own vocabulary, not stock
 * ornament.
 */

type Tile = {
  Icon: typeof FileText;
  /** Percentage coordinates in the field's square coordinate space. */
  x: number;
  y: number;
  size: number;
  accent?: boolean;
  delay: number;
  dur: number;
  opacity: number;
};

const TILES: Record<string, Tile> = {
  document: { Icon: FileText, x: 26, y: 20, size: 56, delay: 0, dur: 12, opacity: 1 },
  mark: { Icon: Highlighter, x: 55, y: 45, size: 68, accent: true, delay: 1.6, dur: 14, opacity: 1 },
  shield: { Icon: ShieldCheck, x: 79, y: 73, size: 52, delay: 3.1, dur: 13, opacity: 0.95 },

  read: { Icon: ScanText, x: 53, y: 13, size: 36, delay: 0.8, dur: 11, opacity: 0.7 },
  quote: { Icon: Quote, x: 84, y: 27, size: 34, delay: 2.4, dur: 15, opacity: 0.6 },
  check: { Icon: SearchCheck, x: 13, y: 50, size: 40, delay: 1.1, dur: 12.5, opacity: 0.75 },
  scale: { Icon: Scale, x: 33, y: 70, size: 34, delay: 3.6, dur: 13.5, opacity: 0.6 },
  unsure: { Icon: CircleHelp, x: 91, y: 52, size: 32, delay: 2.0, dur: 11.5, opacity: 0.55 },
  steps: { Icon: ListChecks, x: 62, y: 90, size: 36, delay: 4.2, dur: 14.5, opacity: 0.6 },
  score: { Icon: Percent, x: 20, y: 87, size: 30, delay: 2.8, dur: 12, opacity: 0.5 },
};

/** Which tiles are drawn as connected. Order is irrelevant; each pair is a line. */
const THREADS: [keyof typeof TILES, keyof typeof TILES][] = [
  ["document", "read"],
  ["document", "mark"],
  ["mark", "quote"],
  ["mark", "check"],
  ["mark", "shield"],
  ["check", "scale"],
  ["shield", "steps"],
  ["shield", "unsure"],
  ["scale", "score"],
];

function IconField() {
  const tiles = Object.entries(TILES);

  return (
    <div
      aria-hidden
      className="relative mx-auto hidden aspect-square w-full max-w-[30rem] select-none sm:block lg:max-w-[34rem]"
    >
      {/* The threads, behind the tiles. Thin, faint, and only between tiles that
          belong together -- a line to everything would be a cobweb. */}
      <svg
        className="absolute inset-0 h-full w-full"
        viewBox="0 0 100 100"
        fill="none"
      >
        {THREADS.map(([from, to], i) => (
          <line
            key={i}
            x1={TILES[from].x}
            y1={TILES[from].y}
            x2={TILES[to].x}
            y2={TILES[to].y}
            stroke="rgb(255 255 255 / 0.13)"
            strokeWidth={0.28}
          />
        ))}
        {tiles.map(([key, t]) => (
          <circle
            key={key}
            cx={t.x}
            cy={t.y}
            r={0.55}
            fill="rgb(255 255 255 / 0.28)"
          />
        ))}
      </svg>

      {tiles.map(([key, t]) => (
        <span
          key={key}
          className={`glyph absolute flex -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-2xl border backdrop-blur-[2px] ${
            t.accent
              ? "border-white/25 bg-brand-500/25 text-white shadow-[0_12px_36px_-10px_oklch(0.55_0.2_288_/_0.8)]"
              : "border-white/15 bg-white/[0.07] text-white shadow-[0_10px_28px_-16px_rgb(0_0_0/0.8)]"
          }`}
          style={
            {
              left: `${t.x}%`,
              top: `${t.y}%`,
              width: t.size,
              height: t.size,
              opacity: t.opacity,
              animationDelay: `${t.delay}s`,
              "--dur": `${t.dur}s`,
            } as CSSProperties
          }
        >
          <t.Icon style={{ width: t.size * 0.4, height: t.size * 0.4 }} />
        </span>
      ))}
    </div>
  );
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
  const invented = pickInvented(recorded) ?? specimen;

  return (
    <>
      {/* ---------------------------------------------------------------- */}
      <section className="hero-wash relative overflow-hidden">
        {/* The glow the product card sits in. Behind everything, and the only
            reason the right-hand column reads as lit rather than pasted on. */}
        <div className="hero-halo" aria-hidden />

        <div className="relative mx-auto w-full max-w-[86rem] px-5 pb-20 pt-28 sm:px-8 sm:pb-24 sm:pt-32 lg:px-12 lg:pb-28 lg:pt-36">
          <div className="grid items-center gap-14 lg:grid-cols-12 lg:gap-10">
            {/* ---- the claim ---- */}
            <div className="lg:col-span-6">
              <Reveal>
                <span className="inline-flex items-center gap-2.5 rounded-full bg-white/[0.07] px-3.5 py-1.5 text-[0.8125rem] font-medium text-brand-100 ring-1 ring-inset ring-white/15">
                  <span className="relative flex h-2 w-2" aria-hidden>
                    <span className="absolute inline-flex h-full w-full rounded-full bg-pass opacity-70" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-pass" />
                  </span>
                  It marks the words, not the answer
                </span>
              </Reveal>

              <Reveal delay={0.05}>
                <h1 className="t-display mt-7 text-balance text-white">
                  AI writes things the document never said.
                </h1>
              </Reveal>

              <Reveal delay={0.1}>
                <p className="t-lead mt-6 max-w-xl text-brand-100/80">
                  Give an AI a document and ask it a question, and most of the
                  answer will be right. The trouble is the part that isn't — it
                  reads exactly like the part that is. TrustRAG marks the exact
                  words the document does not back up, tells you how sure it is
                  in a number you can rely on, and points at the sentence you
                  should read to settle it.
                </p>
              </Reveal>

              <Reveal delay={0.15}>
                <div className="mt-9 flex flex-wrap gap-3">
                  <Button onClick={onDemo}>
                    See it on a real example
                    <ArrowRight className="h-4 w-4" aria-hidden />
                  </Button>
                  <Button variant="onDark" onClick={onProof}>
                    Why believe it
                  </Button>
                </div>
              </Reveal>

              {/* The proof, above the fold. Three measured numbers, each one a
                  claim the evidence page then has to defend. */}
              <Reveal delay={0.2}>
                <dl className="mt-12 grid max-w-lg grid-cols-3 gap-6 border-t border-white/10 pt-7">
                  {SCORECARD.map((item) => (
                    <div key={item.title}>
                      <dt className="sr-only">{item.title}</dt>
                      <dd>
                        <span className="tnum block text-[1.75rem] font-semibold leading-none tracking-tight text-white">
                          {item.value}
                          <span className="ml-1 text-sm font-medium text-brand-200/70">
                            {item.unit}
                          </span>
                        </span>
                        <span className="mt-2 block text-[0.8125rem] leading-5 text-brand-100/60">
                          {HERO_STAT_LABEL[item.title] ?? item.title}
                        </span>
                      </dd>
                    </div>
                  ))}
                </dl>
              </Reveal>
            </div>

            {/* ---- the field ---- */}
            <div className="lg:col-span-6">
              <Reveal delay={0.12}>
                <IconField />
              </Reveal>
            </div>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------------- */}
      <Band tone="white">
        <BandHead
          eyebrow="The problem"
          title="It looks exactly like the truth."
          lead={
            invented
              ? "The one above had a sentence in the document to check it against. This one has none. Same system, a different real answer — and the document simply never mentions it."
              : "An AI is given a document and asked to summarise it. Most of what comes back is fine. One clause is not."
          }
        />

        {invented ? (
          <Reveal>
            <div className="grid gap-6 lg:grid-cols-2">
              <Card className="p-6 sm:p-8">
                <p className="t-eyebrow text-slate-500">What the AI wrote</p>
                <p className="t-answer mt-4 text-slate-800">
                  {(() => {
                    // Shown in its sentence rather than on its own. A marked
                    // fragment with nothing either side of it is unreadable, and
                    // a reader cannot judge a claim they cannot parse.
                    const { lead, marked, tail } = excerpt(
                      invented.result.answer,
                      invented.span,
                      110,
                      110,
                    );
                    return (
                      <>
                        <span className="text-slate-500">{lead}</span>
                        <span className={`mark ${VERDICT_SKIN.flag.mark}`}>{marked}</span>
                        <span className="text-slate-500">{tail}</span>
                      </>
                    );
                  })()}
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
                  {invented.span.evidence_sentence
                    ? "What the document actually says"
                    : "What the document says about it"}
                </p>
                <p className="t-lead mt-4 text-slate-700">
                  {invented.span.evidence_sentence ? (
                    <>
                      <Quote className="mr-2 inline h-4 w-4 -translate-y-0.5 text-slate-400" aria-hidden />
                      {invented.span.evidence_sentence}
                    </>
                  ) : (
                    "Nothing. We read it one sentence at a time and not one of them backs this up."
                  )}
                </p>
                <p className="t-body mt-6 text-slate-500">
                  A person who had not read the document would never have caught
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
