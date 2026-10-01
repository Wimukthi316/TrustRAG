// Every word a person reads lives here.
//
// The rule, and it is the whole point of this file: **nothing a reader sees may
// be a word from inside the machine.** The people this is built for are not the
// people who built it. "Span", "conformal", "calibrated", "coverage",
// "miscoverage", "alpha", "entailment", "attribution", "non-conformity" and
// "exchangeability" are all words this codebase uses about itself, and not one
// of them tells a person whether they can trust a sentence.
//
// There is no longer a switch that reveals the technical register. It existed,
// and it was doing nothing except splitting every screen into two versions that
// both had to be maintained. The technical account of this system lives in the
// paper, where it belongs.

import type { ConformalDecision, ErrorType, TaskType } from "./types";

export const VERDICT = {
  flag: {
    label: "Not in the document",
    short: "Not in the document",
    lead: "The document does not say this.",
  },
  abstain: {
    label: "Not sure",
    short: "Not sure",
    lead: "We could not tell either way, so we are not going to guess. A person should look at this one.",
  },
  pass: {
    label: "In the document",
    short: "In the document",
    lead: "The document backs this up.",
  },
} satisfies Record<
  ConformalDecision,
  { label: string; short: string; lead: string }
>;

// What the evidence panel says, in the cases that matter.
export const EVIDENCE = {
  foundTitle: "The document says this instead",
  foundTitleNeutral: "This is the part of the document we checked",
  noneTitle: "Nothing in the document says this",
  noneBody:
    "We read the document one sentence at a time and none of them backs this up. It was not in what the AI was given.",
  notRunTitle: "We did not check the document for this one",
  notRunBody:
    "This result came back without the document check attached, so we are not saying anything either way about where the evidence is.",
} as const;

export const SURE = {
  label: "How sure we are it is made up",
  hint: "Checked against thousands of answers we already knew the truth about. When it says 80, it really is wrong about 80 times in 100.",
} as const;

// What the system would and would not rule out, for one passage.
//
// The decision is a **set**, not a label. The machinery keeps every answer it
// cannot rule out at the chosen level of care, and the three verdicts on
// screen are only the three shapes that set can take: one answer left, the
// other answer left, or both. Showing the set is the one place a reader can
// see that "not sure" is a decision taken deliberately rather than a number
// that happened to land in the middle -- which is the single most common
// misreading of this system.
export const RULED = {
  // The plain sentence is the heading and the chips are the evidence under
  // it, not the other way round. "What it would not rule out" is accurate and
  // it is also a double negative, which is not a thing to lead with.
  both: "It could go either way, and it will not guess. Read this one yourself.",
  none: "At this setting it will not stand behind either answer for this passage.",
  title: "What it could not rule out",
  hint: "The system keeps every answer it cannot rule out at the current setting. When both survive it stops and says so, rather than reporting whichever number happened to be larger.",
  yes: "In the document",
  no: "Not in the document",
} as const;

// C2's correction, shown on the passage it was applied to.
//
// The reliability diagram on the proof page argues this in aggregate, over
// 2,390 held-out passages, which is the right way to argue it and the wrong
// way to show it to somebody watching a demo. Here it is one number moving:
// what the detector said, and what it is worth once the correction is applied.
// Without this the calibration layer is invisible on the screen where the
// system is actually being judged.
export const CORRECTION = {
  label: "Before correcting",
  moved:
    "The detector's own score is the first number. The second is what that score is worth once it has been corrected against thousands of answers whose truth we already knew.",
  same:
    "The correction left this one where it was. It does not move every score -- it moves the ones the detector was over-confident about.",
} as const;

// The distribution check, opened up.
//
// It shipped as a single green or amber card, which asks the reader to take
// the verdict on trust -- from the one component whose entire purpose is not
// asking anyone to take things on trust. The per-feature numbers were already
// being computed and sent on every response and nothing read them.
export const FIT = {
  title: "Is this the kind of text it was tested on?",
  lead: "Eight things it measured about this text, each shown against the range it saw while being tested. None of them is about whether the answer is true — only about whether the odds above were worked out on material like this.",
  typical: "Ordinary",
  unusual: "Unlike the test material",
  low: "smaller than almost all of them",
  high: "larger than almost all of them",
  middle: "in the ordinary range",
  onesided:
    "This can only ever find a difference, never rule one out. It looks at a handful of quantities, and a text can differ in a way none of them measures.",
} as const;

// The guarantee for one kind of text, rather than the pooled one.
//
// Coverage is reported per task in the results and only the pooled figure ever
// reached the screen. Pooled is the weaker claim: a system can hold 88% overall
// while missing badly on one of three tasks, and a reader looking at a
// summarisation result deserves the summarisation number.
export const TASK_PROMISE = {
  title: "And for this kind of text on its own",
  lead: "The promise above is the figure across everything it was tested on. Broken out, this is how it did on texts like the one on screen.",
  asked: "Asked to catch",
  caught: "Caught",
  inside: "Inside the range a fair run is allowed to land in.",
  outside: "Outside that range. This one did not hold, and it is here rather than hidden.",
  none: "This kind of text was not broken out separately, so only the overall figure applies.",
} as const;

export const TASK_LABEL: Record<TaskType, string> = {
  qa: "Answering a question",
  summarization: "Shortening a long document",
  data2text: "Writing up a record",
  other: "Something else",
};

export const ERROR_TYPE_LABEL: Record<ErrorType, string> = {
  evident_conflict: "Says the opposite of the document",
  subtle_conflict: "Quietly disagrees with the document",
  evident_baseless_info: "Made up",
  subtle_baseless_info: "Extra detail nobody gave it",
  unknown: "Not sorted",
};

// The four categories are not four unrelated labels. They are two questions
// asked separately, and the system is much better at one of them than the
// other, so a reader is told them separately rather than being handed one
// compound phrase like "Extra detail nobody gave it" and left to unpack it.
//
//   mechanism -- does it contradict the document, or is the document simply
//                silent about it? This is the reliable axis.
//   severity  -- would a person notice on one read, or does it slip past?
//                This is the weak axis, and the system frequently declines to
//                decide it. When it does, the sentence underneath says so.
//
// `family` only drives colour: the two mechanisms are different problems and a
// reader should be able to tell them apart across four cards without reading.
export const ERROR_TYPE_DETAIL: Record<
  ErrorType,
  {
    family: "conflict" | "baseless" | "unknown";
    mechanism: string;
    severity: string;
    meaning: string;
  }
> = {
  evident_conflict: {
    family: "conflict",
    mechanism: "The document says otherwise",
    severity: "Obvious once you look",
    meaning:
      "There is a sentence in the document that this directly contradicts. The answer did not invent the subject — it got it wrong.",
  },
  subtle_conflict: {
    family: "conflict",
    mechanism: "The document says otherwise",
    severity: "Quiet — easy to miss on one read",
    meaning:
      "The wording follows the document closely and one detail has been changed. This is the kind that survives proofreading.",
  },
  evident_baseless_info: {
    family: "baseless",
    mechanism: "The document never mentions this",
    severity: "Obvious once you look",
    meaning:
      "Nothing in the document is about this at all. It did not come from the source the answer was supposed to use.",
  },
  subtle_baseless_info: {
    family: "baseless",
    mechanism: "The document never mentions this",
    severity: "Quiet — easy to miss on one read",
    meaning:
      "An extra detail that reads as though it belongs, and the document does not support it. It may even be true — it is just not in the source.",
  },
  unknown: {
    family: "unknown",
    mechanism: "Not sorted",
    severity: "Not sorted",
    meaning: "",
  },
};

export const FIELD = {
  question: {
    label: "What someone asked",
    plain: "Leave this empty if nobody asked anything.",
  },
  context: { label: "The document", plain: "The only thing the answer is allowed to use." },
  answer: { label: "What the AI wrote", plain: "This is the part we check." },
  task: { label: "What the AI was doing", plain: "" },
} as const;

// The one dial, described by what it buys rather than by what it is.
export function careLabel(alpha: number): { name: string; line: string } {
  const catches = Math.round((1 - alpha) * 100);
  if (alpha <= 0.1) {
    return {
      name: "Very careful",
      line: `Catches ${catches} out of every 100. Says "not sure" more often.`,
    };
  }
  if (alpha <= 0.2) {
    return { name: "Careful", line: `Catches ${catches} out of every 100.` };
  }
  return {
    name: "Quick",
    line: `Catches ${catches} out of every 100. Almost always gives an answer.`,
  };
}

// The dial.
//
// It is the only control on the page that changes an answer, and the reason it
// is worth having is not that a reader wants to fiddle with it. It is that the
// trade it makes -- catch more, refuse more -- is the entire argument of this
// component, and an argument you can move with your hand is one you remember.
//
// Every position on it was measured. The figures printed beside it are what
// happened on the held-out split at that exact setting, not a projection from
// the one setting everything else is reported at.
// The plain-language summary, and the reason it exists.
//
// Everything else on this page answers a question somebody who built it would
// ask. This answers the four a person who did not build it actually asks, in
// this order: what do I do now, how much work did that save me, can I trust
// the number, and does any of it apply to what I just pasted in.
//
// It is C2's output and nothing else -- the counts are the conformal
// decisions, the trust line is the calibration, the last line is the
// exchangeability check. No number in it is computed here; they are all
// already on the page in a form only an author could read.
//
// It replaced three stat boxes that printed the same three counts with no
// sentence attached. A count is not an answer to "what do I do now".
export const PLAIN = {
  title: "What this means for you",

  // The row headings are the same three words used in the legend, on the
  // chips and on every card. A summary that invents its own vocabulary makes
  // the reader learn the page twice.
  flagBody:
    "Read these ones. It is confident enough about them to stand behind them.",
  unsureBody:
    "It could not tell either way, so it is handing these to you rather than picking the likelier answer. Most tools of this kind will not do this.",
  cleanTitle: "Left unmarked",
  cleanBody: "Backed by the document. You do not need to read it again.",

  nothing: "It went through the whole answer and found nothing the document does not support.",

  // The counterweight is not optional. A percentage with nothing to read it
  // against is the habit this project exists to complain about.
  trustWith: (n: number) =>
    `The percentages here are not the model's opinion of itself. They were corrected against ${n.toLocaleString()} answers whose truth was already known, so a passage marked 80 in 100 really is made up about 80 times in 100.`,
  trustWithout:
    "The percentages here are not the model's opinion of itself. They were corrected against thousands of answers whose truth was already known, so a passage marked 80 in 100 really is made up about 80 times in 100.",

  appliesYes: "This text is the kind it was tested on, so those odds apply here.",
  appliesNo:
    "This text is not like anything it was tested on. The marks stand — the odds do not apply here.",
  appliesNone: "No accuracy promise is attached to this one.",

  effort: (sentences: number, marks: number) =>
    `Checking this by hand means reading all ${sentences} sentences of the document. It has narrowed that down to ${marks} ${
      marks === 1 ? "passage" : "passages"
    }.`,
  effortNothing: (sentences: number) =>
    `It read all ${sentences} sentences of the document and found nothing it needed to hand you.`,
} as const;

export const DIAL = {
  title: "How careful should it be?",
  lead: "Turn it up and the system refuses more often and misses less. Turn it down and it commits to an answer almost every time. Watch the marks above change as you move it.",
  // "in 100" rather than "%": a percentage is a unit somebody has to convert
  // before it means anything, and this row is read by people who will not.
  asked: "Set to catch",
  caught: "Actually caught",
  unsure: 'Says "not sure"',
  stands: "Will stand behind",
  perHundred: "in 100",
  // The same four numbers as a sentence, above the four numbers. Somebody who
  // reads only the sentence has still been told the trade.
  summary: (asked: number, caught: number, unsure: number) =>
    `Set to catch ${asked} out of every 100 made-up passages. When that was tested it caught ${caught}. The price is that it says "not sure" about ${unsure} out of every 100 passages it looks at.`,
  inside: "That is inside the range a fair test is allowed to land in.",
  outside: "That is below the range a fair test is allowed to land in, and it says so rather than rounding up.",
  measured: "Measured at this exact setting, not carried over from another one.",
  frozen: "This check was reopened from your history, so the dial shows the setting it was run at. Start a new check to move it.",
  rerunning: "Checking again at the new setting…",
  live: "Moving this re-runs the check, because the setting changes what the system will stand behind.",
} as const;

export const PROMISE = {
  holds: {
    title: "This one is the kind of text we tested on",
    body: "Nothing about it looked unusual, so the percentages above mean what they say.",
  },
  withdrawn: {
    title: "This text is not like anything we tested on",
    body: "The marks below are still the system's. What we are taking back is the promise about how often it is right — not the check itself. Treat the percentages as a guess here.",
  },
  absent: {
    title: "No accuracy promise attached",
    body: "The marks are the model's raw opinion, with nothing behind them.",
  },
} as const;

// The five steps, told as what happens to a paragraph.
export const STEPS = [
  {
    title: "It reads both at once",
    plain:
      "The document the AI was given, and the answer the AI wrote from it, side by side.",
  },
  {
    title: "It marks what is not in there",
    plain:
      "Word by word through the answer, marking the parts the document does not back up. Not whole answers — the exact words.",
  },
  {
    title: "It says how sure it is, honestly",
    plain:
      "A raw score out of a model means nothing on its own. Ours was corrected against thousands of answers we already knew the truth about, so 80% means 80%.",
  },
  {
    title: "If it cannot tell, it says so",
    plain:
      "Anything it cannot separate cleanly is handed to a person instead of being guessed at. Most tools of this kind will not do this.",
  },
  {
    title: "It shows you where to check",
    plain:
      "For every mark it points at the sentence in the document you should read — or tells you there isn't one.",
  },
] as const;

// The three things it can say about a piece of text, for the home page.
export const VERDICT_CARDS = [
  {
    tone: "flag" as const,
    title: "Not in the document",
    body: "The AI wrote it, the document never said it. This is the one you were looking for.",
  },
  {
    tone: "abstain" as const,
    title: "Not sure",
    body: "It could not tell, so it refuses to guess and passes this one to you. Refusing is a feature, not a failure.",
  },
  {
    tone: "pass" as const,
    title: "In the document",
    body: "Left unmarked. The document backs it up and you do not need to read it again.",
  },
] as const;

// The failures, in the same plain register as everything else.
export const FAULTS = [
  {
    title: "A bigger tool beats it, and we say so",
    body: "The best public tool of this kind scores 79 out of 100 on this task. Ours scores 76, because theirs is a larger model.",
  },
  {
    title: "On a different kind of document, the promise breaks",
    body: "Take it to writing it was not tested on and it stops keeping its promise. We tried three ways to fix that and none of them worked, so instead it checks every text it is given and tells you when the promise does not apply.",
  },
  {
    title: "Half the human examples were not exact quotes",
    body: "The people who marked up the training data reworded the source when they quoted it, so anything judged on exact matching is judged against a moving target.",
  },
  {
    title: "On one kind of document, a silly rule beats it",
    body: "For structured records, always picking the longest line scores higher than our real matcher. Those documents are data rather than writing, so the idea of a 'sentence' barely applies.",
  },
  {
    title: "The fast first pass throws away one in five",
    body: "A quick filter narrows the document down before the careful check runs, and it drops the right sentence 22% of the time. A simpler filter would have dropped 2%. That was our mistake, we measured it, and it is here.",
  },
] as const;

// The headline measurements, said the way a person would say them. Each one
// keeps the honest counterweight, because a number with its floor removed is
// the thing this project exists to complain about.
export const SCORECARD = [
  {
    value: "76",
    unit: "/ 100",
    title: "How often it gets a whole answer right",
    floor:
      "Guessing 'everything is wrong' every time would score 52. The best public tool of this kind scores 79, and it is a bigger model than ours.",
  },
  {
    value: "9",
    unit: "in 10",
    title: "When it makes a promise, it keeps it",
    floor:
      "Asked to catch 9 out of every 10, it caught 8.8 — inside the range a fair run is allowed to land in. It was asked at six different levels and kept all six.",
  },
  {
    value: "3.6",
    unit: "× better",
    title: "Its percentages mean what they say",
    floor:
      "Straight out of the model it was badly over-confident. Corrected against held-out data, '80% sure' now really does mean about 80 in 100.",
  },
] as const;
