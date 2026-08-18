import type {
  ButtonHTMLAttributes,
  HTMLAttributes,
  InputHTMLAttributes,
  LabelHTMLAttributes,
} from "react";

type ButtonVariant = "primary" | "secondary" | "ghost";

export function Button({
  className = "",
  variant = "secondary",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant }) {
  const base =
    "inline-flex min-h-11 items-center justify-center gap-2 rounded-md px-4 py-2 text-sm font-semibold transition duration-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 disabled:opacity-55";
  const styles = {
    primary:
      "bg-[var(--accent)] text-slate-950 hover:bg-[color-mix(in_srgb,var(--accent)_86%,white)] focus-visible:outline-[var(--accent)]",
    secondary:
      "border border-[var(--border)] bg-[var(--surface-raised)] text-[var(--foreground)] hover:border-[var(--border-strong)] hover:bg-[var(--surface-soft)] focus-visible:outline-[var(--primary)]",
    ghost:
      "text-[var(--muted-strong)] hover:bg-[var(--surface-soft)] hover:text-[var(--foreground)] focus-visible:outline-[var(--primary)]",
  } satisfies Record<ButtonVariant, string>;

  return <button className={`${base} ${styles[variant]} ${className}`} {...props} />;
}

export function SectionPanel({
  className = "",
  ...props
}: HTMLAttributes<HTMLElement>) {
  return (
    <section
      className={`panel p-5 md:p-6 ${className}`}
      {...props}
    />
  );
}

export function TextInput({
  className = "",
  ...props
}: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={`control w-full rounded-md px-3 py-2 text-sm ${className}`}
      {...props}
    />
  );
}

export function FieldLabel({
  className = "",
  ...props
}: LabelHTMLAttributes<HTMLLabelElement>) {
  return (
    <label
      className={`mb-2 block text-sm font-semibold text-[var(--muted-strong)] ${className}`}
      {...props}
    />
  );
}

export function Badge({
  className = "",
  ...props
}: HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={`chip inline-flex min-h-7 items-center rounded-full px-2.5 text-xs font-semibold ${className}`}
      {...props}
    />
  );
}

export function SkeletonLine({ className = "" }: { className?: string }) {
  return (
    <div
      className={`h-3 rounded-full bg-[color-mix(in_srgb,var(--border)_70%,transparent)] motion-safe:animate-pulse ${className}`}
    />
  );
}
