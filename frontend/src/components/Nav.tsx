import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { Menu, ShieldCheck, X } from "lucide-react";

export type View = "home" | "demo" | "proof" | "history";

const ITEMS: { id: View; label: string }[] = [
  { id: "home", label: "How it works" },
  { id: "demo", label: "See it work" },
  { id: "proof", label: "Why believe it" },
  { id: "history", label: "Your history" },
];

/**
 * The nav bar.
 *
 * Four destinations and nothing else. It carried a build identifier and a
 * technical-detail switch before; both were removed, because neither answered a
 * question anyone in the room was going to ask.
 *
 * It floats: a pill inset from all three edges rather than a strip pinned to
 * the top. Over the hero it has no background at all, so the page opens as one
 * image instead of as a header with a picture underneath it; the moment the
 * reader scrolls, it fades up a glass panel so the text passing beneath it
 * stays readable. Fixed rather than sticky, because a floating element that
 * leaves a gap above it has to sit over the page, not inside its flow.
 */
export default function Nav({
  view,
  onView,
  overDark = false,
}: {
  view: View;
  onView: (next: View) => void;
  /** True while the bar is floating over the home page's dark hero. */
  overDark?: boolean;
}) {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    function onScroll() {
      setScrolled(window.scrollY > 12);
    }
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  function go(next: View) {
    setOpen(false);
    onView(next);
  }

  // On the home page the bar floats over the dark hero until the reader
  // scrolls, so its ink has to invert with it. Anywhere else it is a normal
  // light bar from the first pixel.
  const dark = overDark && !scrolled && !open;

  return (
    <header className="fixed inset-x-0 top-0 z-50">
      <div className="mx-auto max-w-[86rem] px-4 pt-3 sm:px-6 sm:pt-4 lg:px-10">
        <div
          className={`flex items-center justify-between rounded-2xl px-3 py-2 transition-all duration-300 sm:px-4 ${
            dark
              ? "border border-transparent bg-transparent"
              : open
                ? "glass-light rounded-b-none"
                : "glass-light"
          }`}
        >
          <button
            type="button"
            onClick={() => go("home")}
            className="flex items-center gap-3 rounded-full pr-3 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-brand-600"
          >
            <span
              className={`flex h-9 w-9 items-center justify-center rounded-lg ${
                dark ? "bg-white/15 text-white" : "bg-brand-600 text-white"
              }`}
            >
              <ShieldCheck className="h-[1.125rem] w-[1.125rem]" aria-hidden />
            </span>
            <span>
              <span
                className={`block text-[0.9375rem] font-semibold tracking-tight ${
                  dark ? "text-white" : "text-slate-900"
                }`}
              >
                TrustRAG
              </span>
              <span
                className={`block text-[0.75rem] leading-4 ${
                  dark ? "text-brand-100/70" : "text-slate-500"
                }`}
              >
                Checks AI answers against the document
              </span>
            </span>
          </button>

          <nav className="hidden items-center gap-1 md:flex" aria-label="Sections">
            {ITEMS.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => go(item.id)}
                aria-current={view === item.id ? "page" : undefined}
                className={`relative rounded-full px-3.5 py-1.5 text-[0.875rem] font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 ${
                  dark
                    ? view === item.id
                      ? "text-white"
                      : "text-brand-100/75 hover:text-white"
                    : view === item.id
                      ? "text-slate-900"
                      : "text-slate-600 hover:text-slate-900"
                }`}
              >
                {view === item.id && (
                  <motion.span
                    layoutId="nav-active"
                    transition={{ type: "spring", stiffness: 420, damping: 36 }}
                    className={`absolute inset-0 rounded-full ${
                      dark ? "bg-white/15" : "bg-slate-900/[0.06]"
                    }`}
                  />
                )}
                <span className="relative">{item.label}</span>
              </button>
            ))}
          </nav>

          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-label={open ? "Close menu" : "Open menu"}
            className={`flex h-10 w-10 items-center justify-center rounded-lg ring-1 ring-inset md:hidden ${
              dark ? "text-white ring-white/30" : "text-slate-700 ring-slate-300"
            }`}
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>

        {open && (
          <nav
            className="glass-light -mt-px rounded-b-2xl border-t-0 px-3 py-2 md:hidden"
            aria-label="Sections"
          >
            {ITEMS.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => go(item.id)}
                className={`block w-full rounded-lg px-4 py-3 text-left text-[0.9375rem] font-medium ${
                  view === item.id ? "bg-brand-500/10 text-brand-700" : "text-slate-700"
                }`}
              >
                {item.label}
              </button>
            ))}
          </nav>
        )}
      </div>
    </header>
  );
}
