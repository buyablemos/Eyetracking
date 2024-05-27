const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  await page.goto('http://comarch.pl');
  await page.setViewport({
    width: 1440,
    height:761
  });
  await page.screenshot({ path: 'full_page_screenshot.png', fullPage: true });
  await browser.close();
  process.exit();
})();
