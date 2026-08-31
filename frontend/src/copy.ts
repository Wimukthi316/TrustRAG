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
