import { Component, type ErrorInfo, type ReactNode } from "react";

/**
 * The last line of defence for a demo.
 *
 * Without this, one runtime error anywhere in the tree unmounts everything and
 * leaves a white page — which in front of an audience is indistinguishable from
 * the server being down, and gives nobody anything to act on. With it, the page
 * says what broke and how to get back, and the rest of the machine (the API, the
 * recorded runs) is still there.
 *
 * It deliberately shows the real error text rather than a friendly nothing. The
 * person looking at this screen is the person who has to fix it.
 */
interface State {
  error: Error | null;
}

export default class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Kept on the console as well, because the stack is more useful there than
    // anything this component could render.
    console.error("TrustRAG interface crashed:", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="flex min-h-full items-center justify-center bg-slate-50 p-6">
        <div className="max-w-lg rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <h1 className="text-base font-semibold text-slate-900">
            The interface stopped, and the detector did not.
          </h1>
          <p className="mt-2 text-sm leading-6 text-slate-600">
            Something in the page threw an error. The API is a separate process
            and is almost certainly still running — you can check it directly at{" "}
            <code className="rounded bg-slate-100 px-1 py-0.5 text-xs">
              http://127.0.0.1:8000/docs
            </code>
            .
          </p>
          <pre className="mt-4 max-h-48 overflow-auto rounded-lg bg-slate-900 p-3 text-[11px] leading-5 text-slate-100">
            {error.message}
          </pre>
          <div className="mt-4 flex gap-2">
            <button
              onClick={() => this.setState({ error: null })}
              className="rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-slate-700"
            >
              Try again
            </button>
            <button
              onClick={() => window.location.reload()}
              className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-medium text-slate-700 transition hover:bg-slate-50"
            >
              Reload the page
            </button>
          </div>
        </div>
      </div>
    );
  }
}
