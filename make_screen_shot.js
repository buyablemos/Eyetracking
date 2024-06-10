const puppeteer = require('puppeteer');

// Pobieranie argumentów z linii poleceń
const args = process.argv.slice(2);
const url = args[0];  // Pierwszy argument to URL
const width = parseInt(args[1], 10);  // Drugi argument to szerokość viewportu
const height = parseInt(args[2], 10);  // Trzeci argument to wysokość viewportu

if (!url || isNaN(width) || isNaN(height)) {
  console.error('URL, width, and height are required as arguments');
  process.exit(1);
}

(async () => {
  const browser = await puppeteer.launch({
    args: ['--ignore-certificate-errors']
  });
  const page = await browser.newPage();
  await page.goto(url);
  await page.setViewport({
    width: width,
    height: height
  });

  await new Promise(resolve => setTimeout(resolve, 8000));

  await page.screenshot({ path: 'full_page_screenshot.png', fullPage: true });
  await browser.close();
  console.log(`Screenshot of ${url} taken with viewport ${width}x${height} and saved as full_page_screenshot.png`);
  process.exit();
})();
