export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        wash: '#EFF4FA',
        ink: '#0F172A',
        body: '#475569',
        navy: '#020817',
        accent: { DEFAULT: '#2563EB', bright: '#3B82F6' },
      },
      fontFamily: {
        display: ['Sora', 'system-ui', 'sans-serif'],
        body: ['system-ui', '-apple-system', '"Segoe UI"', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
