import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ScholarNavigator",
  description: "复杂学术查询的智能论文搜索与推荐系统",
  icons: {
    icon: [{ url: "/assets/scholarnavigator-compass-logo-black.png", type: "image/png" }],
    apple: [{ url: "/assets/scholarnavigator-compass-logo-black.png", type: "image/png" }],
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" className="dark">
      <body>{children}</body>
    </html>
  );
}
