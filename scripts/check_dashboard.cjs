// Optional browser check; requires Playwright and a local Chromium executable.
// Start the dashboard on port 8765, then: node scripts/check_dashboard.cjs
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const http = require('node:http');
const fs = require('node:fs');

(async () => {
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium', headless: true, args:['--no-sandbox']});
  const page = await browser.newPage({viewport:{width:1440,height:1150}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const base = process.env.IOV_DEMO_URL || 'http://127.0.0.1:8765';
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('Fresh rollout'));
  const trained = await page.locator('#reward').textContent();
  assert.equal(await page.locator('#vehicles .vehicle').count(), 6);
  await page.locator('#policy').selectOption('untrained');
  await page.waitForFunction(old => document.querySelector('#reward').textContent !== old, trained);
  assert.notEqual(await page.locator('#reward').textContent(), trained);
  await page.locator('#policy').selectOption('trained');
  await page.waitForFunction(old => document.querySelector('#reward').textContent === old, trained);
  await page.locator('#workload').selectOption('1.5');
  await page.locator('#seed').fill('310123');
  await page.locator('#simulate').click();
  await page.waitForFunction(() => document.querySelector('#status').textContent.includes('seed 310123'));
  await page.locator('#scrub').fill('10');
  assert.equal(await page.locator('#clock').textContent(), '11 / 40');
  await page.locator('#play').click();
  await page.waitForFunction(() => document.querySelector('#clock').textContent !== '11 / 40');
  await page.locator('#play').click();
  await page.locator('#workload').selectOption('1');
  await page.locator('#seed').fill('300000');
  await page.locator('#simulate').click();
  await page.waitForFunction(() => document.querySelector('#status').textContent.includes('seed 300000'));
  await page.screenshot({path:'artifacts/demo/dashboard.png',fullPage:true});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({path:'artifacts/demo/dashboard-mobile.png',fullPage:true});
  const health = await page.request.get(base+'/api/health');
  assert.equal((await health.json()).model_loaded,true);
  const invalid = await page.request.get(base+'/api/rollout?workload=5');
  assert.equal(invalid.status(),400);
  const missing = await page.request.get(base+'/does-not-exist');
  assert.equal(missing.status(),404);
  // Serve the single export as a normal local document: managed Chromium blocks
  // file: navigation. Disable networking after load to verify standalone replay.
  const offlineServer = http.createServer((request,response) => {
    response.setHeader('Content-Type','text/html; charset=utf-8');
    response.end(fs.readFileSync(path.resolve('artifacts/demo/showcase.html')));
  });
  await new Promise(resolve => offlineServer.listen(0,'127.0.0.1',resolve));
  await page.goto('http://127.0.0.1:'+offlineServer.address().port);
  await page.waitForFunction(() => document.querySelector('#status').textContent.startsWith('Recorded rollout'));
  await page.context().setOffline(true);
  assert.equal(await page.locator('#seed').isDisabled(),true);
  await page.locator('#policy').selectOption('matching');
  await page.waitForFunction(() => document.querySelector('#simulate').disabled === false);
  assert.equal(await page.locator('#status.error').count(),0);
  assert.deepEqual(errors,[]);
  await new Promise(resolve => offlineServer.close(resolve));
  await browser.close();
  console.log('PASS: live inference, trained/untrained switch, custom traffic/load, playback, scrubbing, desktop/mobile layout, HTTP errors, offline replay, and no browser script errors.');
})().catch(error => {console.error(error); process.exit(1)});
