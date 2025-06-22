/** @type {import('tailwindcss').Config} */
module.exports = {
darkMode: 'class', // Enables dark mode via a 'dark' class
content: ["./frontend/src/**/*.{js,jsx,ts,tsx,html}"],
  theme: {
    extend: {
      fontSize: {
        'custom-sm': '0.75rem',  // 12px
        'custom-md': '1rem',      // 16px
        'custom-lg': '1.125rem',  // 20px
      },
    },
  },
  plugins: [],
}

