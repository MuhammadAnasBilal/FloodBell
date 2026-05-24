/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          dark: '#0a0f1d',
          card: 'rgba(15, 23, 42, 0.85)',
          primary: '#3b82f6',
          accent: '#0ea5e9'
        }
      }
    },
  },
  plugins: [],
}
