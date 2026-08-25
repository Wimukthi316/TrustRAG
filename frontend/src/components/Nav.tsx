import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { Menu, ShieldCheck, X } from "lucide-react";

export type View = "home" | "demo" | "proof";

const ITEMS: { id: View; label: string }[] = [
  { id: "home", label: "How it works" },
  { id: "demo", label: "See it work" },
  { id: "proof", label: "Why believe it" },
];

/**
 * The nav bar.
 *
 * Three destinations and nothing else. It carried a build identifier and a
 * technical-detail switch before; both were removed, because neither answered
 * a question anyone in the room was going to ask.
 *
 * It goes translucent-on-scroll rather than being permanently opaque, so the
 * hero reads as a full page rather than as a page with a strip cut off it.
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
    <header
      className={`sticky top-0 z-50 transition-colors duration-300 ${
        dark
          ? "border-b border-transparent bg-transparent"
          : "border-b border-slate-200/80 bg-white/85 backdrop-blur-xl"
      }`}
    >
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4 sm:px-10">
        <button
          type="button"
          onClick={() => go("home")}
          className="flex items-center gap-3 rounded-full pr-3 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-brand-600"
        >
          <span
            className={`flex h-10 w-10 items-center justify-center rounded-xl ${
              dark ? "bg-white/15 text-white" : "bg-brand-600 text-white"
            }`}
          >
            <ShieldCheck className="h-5 w-5" aria-hidden />
          </span>
          <span>
            <span
              className={`block text-[1.0625rem] font-semibold tracking-tight ${
                dark ? "text-white" : "text-slate-900"
              }`}
            >
              TrustRAG
            </span>
            <span
              className={`block text-[0.8125rem] leading-4 ${
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
              className={`relative rounded-full px-4 py-2 text-[0.9375rem] font-medium transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 ${
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
          className={`flex h-11 w-11 items-center justify-center rounded-xl ring-1 ring-inset md:hidden ${
            dark ? "text-white ring-white/30" : "text-slate-700 ring-slate-300"
          }`}
        >
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </div>

      {open && (
        <nav
          className="border-t border-slate-200 bg-white px-6 py-3 md:hidden"
          aria-label="Sections"
        >
          {ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => go(item.id)}
              className={`block w-full rounded-xl px-4 py-3.5 text-left text-[1.0625rem] font-medium ${
                view === item.id
                  ? "bg-brand-50 text-brand-700"
                  : "text-slate-700"
              }`}
            >
              {item.label}
            </button>
          ))}
        </nav>
      )}
    </header>
  );
}
