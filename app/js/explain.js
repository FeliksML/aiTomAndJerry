/* "How it works" — the six-step explainer, one set per school.
 *
 * The other explainer in this app (`l`, panels.js) is a live readout: real telemetry
 * from the optimiser that is running right now. This one is the opposite and they are
 * deliberately both here. Nothing on these screens is measured. It is the *idea* of the
 * algorithm, drawn once, advanced by hand, so the narration can sit on a beat as long
 * as it wants and the training behind it is paused rather than racing ahead.
 *
 * Rules the copy follows, and they are the reason it reads the way it does:
 *
 *   - No formulas, no notation, no Greek unless it is the name of a dial the algorithm
 *     genuinely has (sigma).
 *   - Still accurate. The advantage/critic step, the clipped update, selection without
 *     learning, and covariance adaptation are the real mechanisms — stripped of maths,
 *     not replaced by a metaphor that would fall apart under a follow-up question.
 *   - One idea per step, one picture, one line you could repeat in a pub.
 *
 * A sealed school never gets here: the overlay refuses to open and app.js draws the
 * classified plate instead, the same as every other screen that could leak a name.
 *
 * The diagrams are pure CSS animation on a fixed 1190x804 board scaled into the pane,
 * so they cost nothing per frame and record identically every take.
 */
(function (global) {
  'use strict';

  var P = global.Paint;

  /* The same sheets the arena draws from — not a second set of art. A diagram that says
     "48 slightly different Toms" is worth far more when they are the Tom the viewer has
     been watching hunt for the last five minutes.
     Both fall back to the plain shape they replace: the sheets are fetched once at
     startup, long before anyone presses `w`, but a missing PNG must cost a silhouette,
     not a diagram. */
  function cat() { return global.CatSprite; }
  function mouse() { return global.MouseSprite; }

  /* Sized by how tall the CHARACTER should be, never by the frame: both sheets pad the
     cell generously and the padding is not the same on both, so sizing by the frame
     would draw Jerry and Tom at two different scales for the same number. */
  function spriteBox(sheet, direction, frame, charPx, extra) {
    if (!sheet || !sheet.ready()) return null;
    var m = sheet.meta();
    var size = charPx / (m.charHeight || 0.8);
    var b = sheet.backgroundStyle(direction, frame, size);
    return '<div style="width:' + b.width + ';height:' + b.height
      + ';background-image:' + b.backgroundImage + ';background-size:' + b.backgroundSize
      + ';background-position:' + b.backgroundPosition + ';background-repeat:no-repeat'
      + (extra ? ';' + extra : '') + '"></div>';
  }

  /* Placed by the ground point, the way the arena places them: (ax, ay) is where the feet
     stand. Centring the frame instead would float the character above its own spot. */
  function spriteAt(sheet, direction, frame, ax, ay, charPx, extra) {
    if (!sheet || !sheet.ready()) return null;
    var m = sheet.meta();
    var size = charPx / (m.charHeight || 0.8);
    return spriteBox(sheet, direction, frame, charPx,
      'position:absolute;left:' + (ax - size * m.anchor.x).toFixed(1) + 'px;top:'
        + (ay - size * m.anchor.y).toFixed(1) + 'px' + (extra ? ';' + extra : ''));
  }

  function esc(s) {
    return String(s === undefined ? '' : s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  /* ---------------- the copy ---------------- */

  var STEPS = {
    ppo: [
      { k: 'STEP 1 · THE BRAIN', t: 'Tom doesn’t have a plan. He has odds.',
        b: ['His policy is one box. Everything he can see goes in — walls, the cone of vision, the smell of mouse — and out comes a set of odds over the five things he could do.',
            'It never says “go right”. It says “right, 63% of the time”. Then he rolls the dice against his own odds. That leftover randomness isn’t sloppiness — it’s the only way he ever discovers anything new.'],
        take: 'A policy is odds, not orders.' },
      { k: 'STEP 2 · GO PLAY', t: 'Nobody learns anything for the first two thousand steps.',
        b: ['He plays with the brain he already has. Whole episodes: chases, near-misses, a proud faceplant into a trap. Every single step gets written down — what he saw, what he did, what came of it.',
            'Only when the tape is full does training start. Play first, think later.'],
        take: 'Fill the tape, then learn from it.' },
      { k: 'STEP 3 · THE BOOKIE', t: 'There’s a bookie in his head, and he’s the one grading.',
        b: ['Next to the policy sits a second little box with one job: guess the score in advance. “From a spot like this, I usually end up around 40.”',
            'So the question is never “was that good?” It’s “was that better than expected?” Expected 40, got 70 → that move scores +30. Expected 60, got 45 → −15. A win can grade negative if it was an easy win.'],
        take: 'Grade the surprise, not the score.' },
      { k: 'STEP 4 · THE NUDGE', t: 'Surprise turns the dials.',
        b: ['Every action on the tape now carries a number. Positive surprise? Make that action a bit more likely next time. Negative? A bit less likely.',
            'That is the entire learning step. Five odds, gently reweighted, thousands of times.'],
        take: 'Good surprise up, bad surprise down.' },
      { k: 'STEP 5 · THE LEASH', t: 'The P in PPO stands for a leash.',
        b: ['One glorious lucky catch, and a naive learner slams that action from 12% to 80% — torching everything else it knew. PPO refuses. Each update is clipped to a small ring around the old odds: 12% may become 13.5%, never 80%.',
            'It isn’t just caution. The tape was recorded by yesterday’s Tom, so it only describes the world near him. Sprint away and your own data starts lying to you.'],
        take: 'Small steps, because the data belongs to the old you.' },
      { k: 'STEP 6 · THE LOOP', t: 'Then do all of that about five hundred more times.',
        b: ['Play → grade the surprise → nudge → clip → bin the tape → play again with the slightly better brain.',
            'It is deliberately boring, and boring is the whole feature: PPO very rarely has the catastrophic bad day where a model forgets everything overnight.'],
        take: 'Slow, steady, hard to break — that’s why PPO’s Tom hunts like a professional.' }
    ],
    ga: [
      { k: 'STEP 1 · THE CROWD', t: 'Forget one student. Bring 48 slightly different Toms.',
        b: ['No gradients, no bookie, no clever maths. A genetic algorithm keeps a whole population, and each Tom is nothing but a long list of numbers — his genome. Identical wiring, different settings.',
            'At generation 1 those numbers are pure noise. Forty-eight random idiots, each idiotic in his own personal way.'],
        take: 'One student becomes a crowd.' },
      { k: 'STEP 2 · THE EXAM', t: 'Everyone runs the maze. Everyone comes back with one number.',
        b: ['That single score is the whole feedback. Not per action, not per step — per entire life.',
            'Nobody asks why #07 scored 82 and #16 scored 17. The algorithm genuinely does not care.'],
        take: 'One life, one number.' },
      { k: 'STEP 3 · THE CULL', t: 'Top of the list lives. Bottom of the list is deleted.',
        b: ['Sort by score, keep roughly the best quarter, bin the rest. Brutal, and shockingly effective.',
            'The population just got better while not a single Tom learned a thing. Improvement came from who was allowed to have children.'],
        take: 'Selection, not learning.' },
      { k: 'STEP 4 · CROSSOVER', t: 'Two survivors, one kid, numbers mixed.',
        b: ['Take dad’s first half and mum’s second half and staple them into a new genome. If dad cut corners well and mum was patient at the hole, sometimes the kid inherits both.',
            'And sometimes he inherits neither and is a disaster. Fine — next exam sorts him out.'],
        take: 'Recombine what already works.' },
      { k: 'STEP 5 · MUTATION', t: 'Then randomly typo a couple of his numbers.',
        b: ['A tiny random jitter on a few genes. It looks like vandalism and it is the most important line in the algorithm.',
            'Without it, survivors keep breeding with their own cousins; ten generations later the whole population is one identical Tom, stuck forever. Mutation is the only door a genuinely new idea can walk through.'],
        take: 'No typos, no new ideas.' },
      { k: 'STEP 6 · NEXT GENERATION', t: 'New crowd. Repeat a couple hundred times.',
        b: ['The curve doesn’t glide upward, it climbs in steps: long flat plateaus, then somebody is born weird and useful and the whole crowd jumps.',
            'Wasteful, slow, and completely blind — but fearless enough to stumble into strategies a careful gradient would never walk towards.'],
        take: 'Dumb, brutal and oddly creative — GA’s Jerry escapes like a lunatic, and it works.' }
    ],
    cmaes: [
      { k: 'STEP 1 · THE CLOUD', t: 'Don’t guess one Tom. Guess a cloud of Toms.',
        b: ['CMA-ES never holds a single answer. It holds a cloud: a centre — its current best guess — and a spread saying how far out, and in which directions, it is still worth looking.',
            'Every point on that map is one complete Tom. Two things are all it remembers.'],
        take: 'The guess is a cloud, not a point.' },
      { k: 'STEP 2 · SAMPLE', t: 'Sprinkle sixteen candidates out of the cloud and race them.',
        b: ['Each dot is a full set of numbers. Each runs the levels and comes back with one score — the same brutal exam the genetic school uses.',
            'The difference is where the candidates come from. Not random guessing: drawn from the cloud, so every round searches somewhere it has reason to search.'],
        take: 'The cloud decides where to look.' },
      { k: 'STEP 3 · MOVE', t: 'Slide the centre onto the winners.',
        b: ['Take the best few, average them — the very best counting most — and put the new centre there.',
            'The losers aren’t punished or deleted. They just pull with a weight of zero.'],
        take: 'Follow the winners’ centre of gravity.' },
      { k: 'STEP 4 · RESHAPE', t: 'Now reshape the cloud. This is the actual trick.',
        b: ['If the good dots all lay along one diagonal, that direction clearly matters — so stretch the cloud along it, and squeeze the directions that changed nothing.',
            'A few rounds later the cloud is a long thin ellipse pointing straight down the useful valley.'],
        take: 'Learn which directions are worth trying at all.' },
      { k: 'STEP 5 · STEP SIZE', t: 'Sprint on the straight, tiptoe near the target.',
        b: ['One more dial: stride length. Drifting the same way several rounds running? Long slope — take bigger steps. Zig-zagging back and forth? We’re circling the answer — take smaller ones.',
            'It reads its own momentum and decides how brave to be, which is why it needs almost no tuning from a human.'],
        take: 'Momentum sets the stride.' },
      { k: 'STEP 6 · CONVERGE', t: 'The cloud collapses onto an answer.',
        b: ['Round after round it tightens until every sample is basically the same Tom. That is convergence, and the final centre is your trained agent.',
            'Fewer wasted runs than a GA, no gradients at all — but it is happiest on a smooth landscape. On a spiky one, 48 brawling Toms can still take it.'],
        take: 'The elegant one — and the only school that trained both animals well.' }
    ]
  };

  /* ---------------- the scenery ---------------- */

  /* Built once and cached. A re-render mid-read — a frame arriving, a reveal firing —
     must not reshuffle the dots the viewer is currently looking at, so none of this is
     allowed to be random at draw time. */
  var D = null;

  function data() {
    if (D) return D;
    var d = {};
    var s = 1337;
    var rnd = function () { return ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648); };
    var i;

    // Each of the 48 gets its own opacity and its own walk phase, so a crowd of the same
    // sprite still reads as forty-eight individuals rather than one Tom stamped out.
    d.pop = [];
    for (i = 0; i < 48; i++) d.pop.push({ o: (0.4 + rnd() * 0.6).toFixed(2), f: i % 4 });

    d.tape = [
      { tag: 'CAUGHT HER', bg: 'rgba(61,220,132,.12)', color: '#7ee6a8', delay: '0s' },
      { tag: 'LOST HER', bg: 'rgba(255,122,84,.1)', color: '#ff9a72', delay: '.1s' },
      { tag: 'STOOD IN A TRAP', bg: 'rgba(242,181,68,.12)', color: '#f2b544', delay: '.2s' },
      { tag: 'CAUGHT HER', bg: 'rgba(61,220,132,.12)', color: '#7ee6a8', delay: '.3s' }
    ];
    d.frames = [];
    for (i = 0; i < 34; i++) d.frames.push(rnd() > 0.72 ? 'rgba(78,168,255,.55)' : 'rgba(120,150,190,.16)');

    d.adv = [
      { scene: 'He cuts the corner and lands right on her scent', pred: '40', predW: '40%',
        got: '70', gotW: '70%', gotBg: '#4ea8ff', delta: '+30',
        chipBg: 'rgba(61,220,132,.14)', chipBorder: 'rgba(61,220,132,.4)', chipColor: '#7ee6a8',
        note: 'do more of that', delay: '.05s', d1: '.15s', d2: '.4s', d3: '.75s' },
      { scene: 'He camps the hole while she slips out the back', pred: '60', predW: '60%',
        got: '45', gotW: '45%', gotBg: 'rgba(255,122,84,.75)', delta: '−15',
        chipBg: 'rgba(255,122,84,.12)', chipBorder: 'rgba(255,122,84,.4)', chipColor: '#ff9a72',
        note: 'do less of that', delay: '.2s', d1: '.3s', d2: '.55s', d3: '.9s' }
    ];

    // The five odds before and after one update. RIGHT is the action that beat the
    // bookie; everything else gives up a sliver of probability to pay for it.
    d.odds = [['UP', 9, 7], ['DOWN', 11, 8], ['LEFT', 8, 6], ['RIGHT', 63, 71], ['STAY', 9, 8]];
    d.nudge = d.odds.map(function (r, n) {
      var up = r[2] > r[1];
      return { name: r[0], label: up ? '#dceaff' : '#9db1cd', oldW: r[1] + '%', newW: r[2] + '%',
        bg: up ? '#4ea8ff' : 'rgba(78,168,255,.4)', arrow: up ? '↑' : '↓',
        arrowColor: up ? '#7ee6a8' : '#ff9a72', diff: (up ? '+' : '−') + Math.abs(r[2] - r[1]),
        delay: (n * 0.07).toFixed(2) + 's', d2: (0.15 + n * 0.07).toFixed(2) + 's' };
    });

    d.loop = [
      { name: 'PLAY', left: '200px', top: '65px' },
      { name: 'GRADE', left: '335px', top: '200px' },
      { name: 'NUDGE', left: '200px', top: '335px' },
      { name: 'CLIP', left: '65px', top: '200px' }
    ];
    d.curvePpo = [];
    for (i = 0; i < 16; i++) {
      var v = 1 / (1 + Math.exp(-(i - 6.5) / 2.6));
      d.curvePpo.push({ h: Math.round(12 + v * 84) + '%', delay: (i * 0.04).toFixed(2) + 's' });
    }
    d.curveGa = [8, 9, 10, 26, 27, 28, 29, 48, 49, 50, 51, 68, 69, 70, 86, 87, 92, 94]
      .map(function (n, j) { return { h: (n + 8) + '%', delay: (j * 0.035).toFixed(3) + 's' }; });
    d.gaLoop = [
      { num: '01', name: 'RUN EVERYONE', delay: '0s' },
      { num: '02', name: 'RANK BY SCORE', delay: '.06s' },
      { num: '03', name: 'CULL THE BOTTOM', delay: '.12s' },
      { num: '04', name: 'CROSSOVER', delay: '.18s' },
      { num: '05', name: 'MUTATE', delay: '.24s' }
    ];

    var ha = [38, 54, 22, 61, 30, 47, 18, 58, 44, 26, 50, 34];
    var hb = [52, 20, 44, 29, 57, 36, 62, 24, 40, 55, 31, 48];
    var px = function (n) { return Math.round(n * 1.05) + '%'; };
    d.geneA = ha.map(function (h, j) {
      return { h: px(h), cut: j < 6 ? '#3ddc84' : 'rgba(61,220,132,.22)', delay: (j * 0.03).toFixed(2) + 's' };
    });
    d.geneB = hb.map(function (h, j) {
      return { h: px(h), cut: j >= 6 ? '#9af0be' : 'rgba(154,240,190,.2)', delay: (j * 0.03).toFixed(2) + 's' };
    });
    d.kid = ha.slice(0, 6).concat(hb.slice(6)).map(function (h, j) {
      return { h: px(h), bg: j < 6 ? '#3ddc84' : '#9af0be', delay: (0.45 + j * 0.04).toFixed(2) + 's' };
    });
    d.mut = d.kid.map(function (g, j) {
      return (j === 2 || j === 9)
        ? { h: j === 2 ? px(72) : px(15), bg: '#f2b544', delay: (j * 0.04).toFixed(2) + 's',
            glow: '0 0 18px rgba(242,181,68,.6)' }
        : { h: g.h, bg: g.bg, delay: (j * 0.04).toFixed(2) + 's', glow: 'none' };
    });

    var names = ['#07', '#22', '#41', '#03', '#18', '#35', '#11', '#29', '#44', '#16'];
    var scores = [82, 74, 69, 61, 55, 48, 40, 33, 25, 17];
    d.board = names.map(function (n, j) {
      return { name: n, score: String(scores[j]), w: scores[j] + '%',
        bg: j < 4 ? '#3ddc84' : 'rgba(61,220,132,.32)', color: j < 4 ? '#9af0be' : '#7d90ad',
        delay: (j * 0.05).toFixed(2) + 's', d2: (0.1 + j * 0.05).toFixed(2) + 's' };
    });
    d.cull = names.map(function (n, j) {
      var keep = j < 4;
      return { name: n, score: String(scores[j]), w: scores[j] + '%',
        bg: keep ? '#3ddc84' : 'rgba(160,180,210,.2)', color: keep ? '#9af0be' : '#5f7392',
        ring: keep ? '#3ddc84' : 'rgba(130,160,200,.2)',
        tag: keep ? 'KEEP' : 'CUT',
        tagBg: keep ? 'rgba(61,220,132,.14)' : 'rgba(255,122,84,.08)',
        tagBorder: keep ? 'rgba(61,220,132,.4)' : 'rgba(255,122,84,.25)',
        tagColor: keep ? '#7ee6a8' : '#a5806f',
        anim: keep ? 'xUp .3s ' + (j * 0.05).toFixed(2) + 's ease-out both'
                   : 'xDrop .55s ' + (0.35 + (j - 4) * 0.06).toFixed(2) + 's ease-out both' };
    });

    var cx = 210, cy = 220;
    var best = [[30, -22], [58, -44], [14, -8], [78, -58]];
    var rest = [[-70, 30], [-40, -60], [20, 70], [-90, -20], [60, 50], [-20, -95],
                [95, 10], [-55, 80], [35, -90], [80, -20], [-100, -55], [10, 105]];
    d.samples = best.map(function (p, j) {
      return { left: (cx + p[0] - 7) + 'px', top: (cy + p[1] - 7) + 'px', size: '14px',
        bg: '#c9a6ff', glow: '0 0 16px rgba(201,166,255,.9)', delay: (j * 0.05).toFixed(2) + 's' };
    }).concat(rest.map(function (p, j) {
      return { left: (cx + p[0] - 5.5) + 'px', top: (cy + p[1] - 5.5) + 'px', size: '11px',
        bg: 'rgba(169,124,255,.42)', glow: 'none', delay: (0.2 + j * 0.035).toFixed(3) + 's' };
    }));
    d.weights = [
      { name: 'BEST', w: '100%', v: '0.40', delay: '.1s' },
      { name: '2ND', w: '70%', v: '0.28', delay: '.18s' },
      { name: '3RD', w: '48%', v: '0.19', delay: '.26s' },
      { name: '4TH', w: '33%', v: '0.13', delay: '.34s' }
    ];
    d.ridge = [[40, -30], [82, -60], [-32, 24], [124, -90]].map(function (p) {
      return { left: (260 + p[0] - 6.5) + 'px', top: (210 + p[1] - 6.5) + 'px' };
    });
    d.converge = [300, 220, 150, 95, 52].map(function (sz, j) {
      return { s: sz + 'px', m: (-sz / 2) + 'px 0 0 ' + (-sz / 2) + 'px',
        o: (0.9 - j * 0.13).toFixed(2), delay: (j * 0.14).toFixed(2) + 's' };
    });

    D = d;
    return d;
  }

  /* ---------------- small shared pieces ---------------- */

  var GRID26 = 'position:absolute;inset:0;background-image:linear-gradient(rgba(120,160,220,.07) 1px,transparent 1px),linear-gradient(90deg,rgba(120,160,220,.07) 1px,transparent 1px);background-size:26px 26px';
  var GRID40 = 'position:absolute;inset:0;background-image:linear-gradient(rgba(120,160,220,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(120,160,220,.06) 1px,transparent 1px);background-size:40px 40px';
  var CAP = 'font-family:var(--display);font-size:10.5px;letter-spacing:2.4px;color:#5f7392';
  var CAP11 = 'font-family:var(--display);font-size:11px;letter-spacing:2.4px;color:#5f7392';
  var NOTE = 'font-size:15px;line-height:1.5;color:#8fa4c4;text-wrap:pretty';
  var CLOUD = 'width:520px;height:420px;flex:0 0 auto;position:relative;border-radius:16px;background:#080b12;border:1px solid rgba(130,160,200,.14);overflow:hidden';

  function oddsRow(name, pct, hot, delay, dur) {
    return '<div style="display:flex;align-items:center;gap:11px">'
      + '<div style="width:54px;font-family:var(--display);font-size:12px;letter-spacing:1.3px;color:'
      + (hot ? '#dceaff' : '#9db1cd') + '">' + name + '</div>'
      + '<div style="flex:1 1 auto;height:15px;border-radius:8px;background:rgba(255,255,255,.06);overflow:hidden">'
      + '<div style="height:100%;width:' + pct + '%;border-radius:8px;background:'
      + (hot ? '#4ea8ff' : 'rgba(78,168,255,.55)')
      + ';transform-origin:left;animation:xBar ' + dur + ' ' + delay + ' ease-out both"></div></div>'
      + '<div style="width:42px;text-align:right;font-family:var(--mono);font-size:12.5px;font-weight:700;color:'
      + (hot ? '#dceaff' : '#9db1cd') + '">' + pct + '%</div></div>';
  }

  /* A parent genome, drawn static: the half that gets passed on is lit, the half that
     does not is dimmed, which is the whole story of the crossover picture. */
  function geneStrip(list) {
    return list.map(function (g) {
      return '<div style="flex:1 1 0;height:' + g.h + ';border-radius:4px;background:' + g.cut + '"></div>';
    }).join('');
  }

  /* ---------------- the diagrams ---------------- */

  var FIG = {};

  /* PPO 1 — one box in, five odds out. The arrows are the whole point: nothing about
     the policy is a decision, it is a distribution that then gets sampled. */
  FIG.ppo1 = function () {
    return '<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;gap:34px">'
      + '<div style="width:286px;flex:0 0 auto;display:flex;flex-direction:column;align-items:center;gap:14px;animation:xUp .34s ease-out both">'
      + '<div style="' + CAP + '">WHAT HE SEES RIGHT NOW</div>'
      + '<div style="width:286px;height:216px;position:relative;border-radius:13px;background:#0a0d13;border:1px solid rgba(130,160,200,.16);overflow:hidden">'
      + '<div style="' + GRID26 + '"></div>'
      + '<div style="position:absolute;left:64px;top:26px;width:150px;height:140px;background:linear-gradient(115deg,rgba(255,122,84,.4),rgba(255,122,84,0));clip-path:polygon(0% 100%,100% 16%,100% 84%);animation:xPulse 2.6s ease-in-out infinite"></div>'
      // Tom stands at the apex of his own vision cone and faces along it; Jerry is the
      // thing at the far end of it. Both by the feet, so they stand on the floor.
      + (spriteAt(cat(), 'up-right', 0, 52, 194, 54)
         || '<div style="position:absolute;left:30px;top:150px;width:42px;height:42px;border-radius:12px;background:#7e90ad;border:2px solid #ff8a5c;display:flex;align-items:center;justify-content:center;font-family:var(--display);font-size:15px;color:#0b1018">T</div>')
      + (spriteAt(mouse(), 'up-right', 1, 223, 80, 32, 'animation:xFlick 1.9s ease-in-out infinite')
         || '<div style="position:absolute;left:210px;top:52px;width:26px;height:26px;border-radius:50%;background:#d09b6a;border:2px solid #6ee2ff;animation:xFlick 1.9s ease-in-out infinite"></div>')
      + '<div style="position:absolute;right:10px;bottom:8px;font-family:var(--mono);font-size:10px;color:#5f7392">walls · cone · scent</div>'
      + '</div></div>'
      + '<div style="font-size:28px;color:#4ea8ff;flex:0 0 auto;animation:xUp .34s .08s ease-out both">→</div>'
      + '<div style="width:212px;height:212px;flex:0 0 auto;position:relative;border-radius:19px;border:1px solid rgba(78,168,255,.42);background:linear-gradient(180deg,rgba(14,38,68,.92),rgba(8,16,30,.92));display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;animation:xPop .38s .14s ease-out both">'
      + '<div style="display:flex;gap:7px">'
      + '<div style="width:9px;height:9px;border-radius:50%;background:#4ea8ff;animation:xPulse 1.4s ease-in-out infinite"></div>'
      + '<div style="width:9px;height:9px;border-radius:50%;background:#4ea8ff;animation:xPulse 1.4s .2s ease-in-out infinite"></div>'
      + '<div style="width:9px;height:9px;border-radius:50%;background:#4ea8ff;animation:xPulse 1.4s .4s ease-in-out infinite"></div></div>'
      + '<div style="font-family:var(--display);font-size:20px;letter-spacing:2.8px;color:#a6d3ff">POLICY</div>'
      + '<div style="font-size:12px;color:#6d80a0">one box of numbers</div></div>'
      + '<div style="font-size:28px;color:#4ea8ff;flex:0 0 auto;animation:xUp .34s .18s ease-out both">→</div>'
      + '<div style="width:328px;flex:0 0 auto;display:flex;flex-direction:column;gap:12px;animation:xUp .34s .22s ease-out both">'
      + '<div style="' + CAP + '">ODDS FOR THIS ONE STEP</div>'
      + oddsRow('UP', 9, false, '.3s', '.5s')
      + oddsRow('DOWN', 11, false, '.36s', '.5s')
      + oddsRow('LEFT', 8, false, '.42s', '.5s')
      + oddsRow('RIGHT', 63, true, '.48s', '.55s')
      + oddsRow('STAY', 9, false, '.54s', '.5s')
      + '<div style="margin-top:8px;font-size:13px;color:#8fa4c4;line-height:1.45">…and then he rolls the dice against exactly these odds.</div>'
      + '</div></div>';
  };

  /* PPO 2 — four episodes on the tape and 2048 rows of it, with nothing learned yet. */
  FIG.ppo2 = function (d) {
    var cards = d.tape.map(function (e) {
      return '<div style="width:212px;height:250px;border-radius:14px;border:1px solid rgba(130,160,200,.16);background:#0a0d13;display:flex;flex-direction:column;overflow:hidden;animation:xPop .4s ' + e.delay + ' ease-out both">'
        + '<div style="flex:1 1 auto;position:relative">'
        + '<div style="position:absolute;inset:0;background-image:linear-gradient(rgba(120,160,220,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(120,160,220,.06) 1px,transparent 1px);background-size:22px 22px"></div>'
        + '<div style="position:absolute;left:22px;top:26px;right:26px;bottom:44px;border-left:2px dashed rgba(255,138,92,.5);border-top:2px dashed rgba(255,138,92,.5);border-top-left-radius:34px"></div>'
        // Standing at the two ends of the dashed trail rather than beside it: Tom where the
        // episode started, Jerry where it ended, so the line reads as the chase itself.
        + (spriteAt(cat(), 'up', 1, 26, 182, 34)
           || '<div style="position:absolute;left:18px;bottom:32px;width:16px;height:16px;border-radius:5px;background:#7e90ad;border:1.5px solid #ff8a5c"></div>')
        + (spriteAt(mouse(), 'right', 3, 186, 42, 24)
           || '<div style="position:absolute;right:20px;top:20px;width:13px;height:13px;border-radius:50%;background:#d09b6a;border:1.5px solid #6ee2ff"></div>') + '</div>'
        + '<div style="flex:0 0 auto;height:40px;display:flex;align-items:center;justify-content:center;gap:8px;background:' + e.bg + ';border-top:1px solid rgba(130,160,200,.12)">'
        + '<span style="font-family:var(--display);font-size:12px;letter-spacing:1.8px;color:' + e.color + '">' + e.tag + '</span>'
        + '</div></div>';
    }).join('');
    var frames = d.frames.map(function (bg) {
      return '<div style="width:19px;height:26px;border-radius:3px;background:' + bg + '"></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;flex-direction:column;justify-content:center;gap:30px">'
      + '<div style="display:flex;gap:20px;justify-content:center">' + cards + '</div>'
      + '<div style="display:flex;flex-direction:column;gap:11px;align-items:center;animation:xUp .4s .35s ease-out both">'
      + '<div style="' + CAP + '">THE TAPE · EVERY STEP WRITTEN DOWN, NOTHING LEARNED YET</div>'
      + '<div style="display:flex;gap:4px">' + frames + '</div>'
      + '<div style="font-family:var(--mono);font-size:12px;color:#6d80a0">saw · did · got — 2048 rows before anyone touches the brain</div>'
      + '</div></div>';
  };

  /* PPO 3 — the critic. Two scenes, each graded against what the bookie expected. */
  FIG.ppo3 = function (d) {
    var rows = d.adv.map(function (a) {
      return '<div style="display:flex;flex-direction:column;gap:12px;padding:20px 22px;border-radius:15px;border:1px solid rgba(130,160,200,.14);background:rgba(255,255,255,.02);animation:xUp .4s ' + a.delay + ' ease-out both">'
        + '<div style="font-family:var(--display);font-size:13px;letter-spacing:1.6px;color:#cddcf2">' + esc(a.scene) + '</div>'
        + '<div style="display:flex;align-items:center;gap:14px">'
        + '<div style="width:120px;font-family:var(--display);font-size:10.5px;letter-spacing:1.5px;color:#7d90ad">BOOKIE GUESSED</div>'
        + '<div style="flex:1 1 auto;height:13px;border-radius:7px;background:rgba(255,255,255,.05);overflow:hidden">'
        + '<div style="height:100%;width:' + a.predW + ';border-radius:7px;background:rgba(160,180,210,.5);transform-origin:left;animation:xBar .45s ' + a.d1 + ' ease-out both"></div></div>'
        + '<div style="width:44px;text-align:right;font-family:var(--mono);font-size:13px;font-weight:700;color:#9db1cd">' + a.pred + '</div></div>'
        + '<div style="display:flex;align-items:center;gap:14px">'
        + '<div style="width:120px;font-family:var(--display);font-size:10.5px;letter-spacing:1.5px;color:#a6d3ff">ACTUALLY GOT</div>'
        + '<div style="flex:1 1 auto;height:13px;border-radius:7px;background:rgba(255,255,255,.05);overflow:hidden">'
        + '<div style="height:100%;width:' + a.gotW + ';border-radius:7px;background:' + a.gotBg + ';transform-origin:left;animation:xBar .45s ' + a.d2 + ' ease-out both"></div></div>'
        + '<div style="width:44px;text-align:right;font-family:var(--mono);font-size:13px;font-weight:700;color:#dceaff">' + a.got + '</div></div>'
        + '<div style="display:flex;align-items:center;gap:12px"><div style="width:120px"></div>'
        + '<div style="height:34px;padding:0 16px;display:flex;align-items:center;gap:9px;border-radius:9px;background:' + a.chipBg + ';border:1px solid ' + a.chipBorder + ';animation:xSpark .35s ' + a.d3 + ' ease-out both">'
        + '<span style="font-family:var(--mono);font-size:16px;font-weight:700;color:' + a.chipColor + '">' + a.delta + '</span>'
        + '<span style="font-family:var(--display);font-size:10.5px;letter-spacing:1.8px;color:' + a.chipColor + '">SURPRISE</span></div>'
        + '<div style="font-size:13px;color:#8fa4c4">' + esc(a.note) + '</div></div></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;align-items:center;gap:36px">'
      + '<div style="width:230px;flex:0 0 auto;display:flex;flex-direction:column;align-items:center;gap:14px;animation:xSlideL .38s ease-out both">'
      + '<div style="width:150px;height:150px;border-radius:20px;border:1px solid rgba(255,209,102,.34);background:rgba(48,38,12,.5);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px">'
      + '<div style="font-family:var(--display);font-size:38px;color:#ffd166">≈</div>'
      + '<div style="font-family:var(--display);font-size:12px;letter-spacing:2px;color:#ffd166">THE BOOKIE</div></div>'
      + '<div style="font-size:13px;line-height:1.45;color:#8fa4c4;text-align:center">a second little box whose only job is to guess the score in advance</div></div>'
      + '<div style="flex:1 1 auto;display:flex;flex-direction:column;gap:26px">' + rows + '</div></div>';
  };

  /* PPO 4 — the update itself: the same five odds, before in grey and after in blue. */
  FIG.ppo4 = function (d) {
    var rows = d.nudge.map(function (n) {
      return '<div style="display:flex;align-items:center;gap:16px;animation:xUp .34s ' + n.delay + ' ease-out both">'
        + '<div style="width:64px;font-family:var(--display);font-size:13px;letter-spacing:1.4px;color:' + n.label + '">' + n.name + '</div>'
        + '<div style="flex:1 1 auto;position:relative;height:26px;border-radius:9px;background:rgba(255,255,255,.05);overflow:hidden">'
        + '<div style="position:absolute;left:0;top:0;bottom:0;width:' + n.oldW + ';border-radius:9px;background:rgba(160,180,210,.22)"></div>'
        + '<div style="position:absolute;left:0;top:0;bottom:0;width:' + n.newW + ';border-radius:9px;background:' + n.bg + ';transform-origin:left;animation:xBar .6s ' + n.d2 + ' ease-out both"></div></div>'
        + '<div style="width:96px;display:flex;align-items:center;gap:8px;justify-content:flex-end">'
        + '<span style="font-family:var(--display);font-size:17px;color:' + n.arrowColor + '">' + n.arrow + '</span>'
        + '<span style="font-family:var(--mono);font-size:13.5px;font-weight:700;color:' + n.arrowColor + '">' + n.diff + '</span>'
        + '</div></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;flex-direction:column;justify-content:center;gap:26px;padding:0 40px">'
      + '<div style="display:flex;justify-content:space-between;align-items:baseline;animation:xUp .3s ease-out both">'
      + '<div style="' + CAP11 + '">THE SAME FIVE ODDS, AFTER ONE UPDATE</div>'
      + '<div style="font-family:var(--mono);font-size:12px;color:#6d80a0">grey = before · blue = after</div></div>'
      + rows
      + '<div style="margin-top:10px;padding:16px 20px;border-radius:13px;background:rgba(78,168,255,.07);border:1px solid rgba(78,168,255,.2);font-size:15px;line-height:1.5;color:#a6d3ff;animation:xUp .4s .5s ease-out both">'
      + 'Nobody ever told Tom the correct move. He just leans a bit harder into whatever kept beating the bookie’s guess.</div></div>';
  };

  /* PPO 5 — the clip. The bar sprints for 80% and gets yanked back inside the fence. */
  FIG.ppo5 = function () {
    return '<div style="width:100%;height:100%;display:flex;flex-direction:column;justify-content:center;gap:34px;padding:0 30px">'
      + '<div style="' + CAP11 + ';animation:xUp .3s ease-out both">ODDS FOR ONE ACTION, AFTER A SINGLE LUCKY CATCH</div>'
      + '<div style="position:relative;height:78px;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid rgba(130,160,200,.14);overflow:hidden;animation:xUp .34s .06s ease-out both">'
      + '<div style="position:absolute;left:0;top:0;bottom:0;width:12%;background:rgba(160,180,210,.22)"></div>'
      + '<div style="position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,rgba(78,168,255,.85),rgba(78,168,255,.35));animation:xClip 1.5s cubic-bezier(.3,.9,.3,1) both"></div>'
      + '<div style="position:absolute;left:9%;top:0;bottom:0;width:6%;border-left:2px dashed rgba(255,209,102,.75);border-right:2px dashed rgba(255,209,102,.75);background:rgba(255,209,102,.09)"></div>'
      + '<div style="position:absolute;left:12%;top:0;bottom:0;width:2px;background:#e6edf8"></div>'
      + '<div style="position:absolute;left:74%;top:0;bottom:0;width:2px;background:rgba(255,122,84,.5)"></div></div>'
      + '<div style="position:relative;height:78px;font-family:var(--display);font-size:11.5px;letter-spacing:1.4px;animation:xUp .34s .12s ease-out both">'
      + '<div style="position:absolute;left:0;top:0;color:#cddcf2">OLD ODDS · 12%</div>'
      + '<div style="position:absolute;left:0;top:26px;color:#ffd166">CLIP FENCE · AS FAR AS ONE UPDATE MAY GO</div>'
      + '<div style="position:absolute;left:0;top:52px;color:#a6d3ff">WHAT IT ACTUALLY BECOMES · 13.5%</div>'
      + '<div style="position:absolute;left:62%;top:0;color:#ff9a72">WHERE ONE LUCKY EPISODE WANTED IT · 80%</div></div>'
      + '<div style="display:flex;gap:20px;animation:xUp .4s .2s ease-out both">'
      + '<div style="flex:1 1 0;padding:20px 22px;border-radius:14px;border:1px solid rgba(255,209,102,.24);background:rgba(48,38,12,.32);display:flex;flex-direction:column;gap:8px">'
      + '<div style="font-family:var(--display);font-size:11px;letter-spacing:1.8px;color:#ffd166">WHY THE LEASH EXISTS</div>'
      + '<div style="font-size:15px;line-height:1.5;color:#cddcf2">The tape was recorded by <b>yesterday’s</b> Tom. Walk far away from him and it stops describing the world the new Tom is actually in.</div></div>'
      + '<div style="flex:1 1 0;padding:20px 22px;border-radius:14px;border:1px solid rgba(130,160,200,.18);background:rgba(255,255,255,.02);display:flex;flex-direction:column;gap:8px">'
      + '<div style="font-family:var(--display);font-size:11px;letter-spacing:1.8px;color:#8fa4c4">WHAT IT BUYS</div>'
      + '<div style="font-size:15px;line-height:1.5;color:#cddcf2">No single glorious accident can rewrite the whole hunter. The <b>P</b> in PPO is literally “stay proximal — stay close to who you were”.</div>'
      + '</div></div></div>';
  };

  /* PPO 6 — the loop, and the shape of the curve it produces. */
  FIG.ppo6 = function (d) {
    var nodes = d.loop.map(function (l) {
      return '<div style="position:absolute;left:' + l.left + ';top:' + l.top + ';width:118px;height:44px;margin:-22px 0 0 -59px;display:flex;align-items:center;justify-content:center;border-radius:11px;border:1px solid rgba(78,168,255,.4);background:rgba(11,28,52,.95);font-family:var(--display);font-size:12.5px;letter-spacing:1.8px;color:#a6d3ff">' + l.name + '</div>';
    }).join('');
    var bars = d.curvePpo.map(function (c) {
      return '<div style="flex:1 1 0;height:' + c.h + ';border-radius:5px 5px 0 0;background:linear-gradient(180deg,#4ea8ff,rgba(78,168,255,.24));transform-origin:bottom;animation:xRise .5s ' + c.delay + ' ease-out both"></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;align-items:center;gap:44px">'
      + '<div style="width:400px;height:400px;flex:0 0 auto;position:relative;animation:xPop .4s ease-out both">'
      + '<div style="position:absolute;left:50%;top:50%;width:270px;height:270px;margin:-135px 0 0 -135px;border-radius:50%;border:1.5px dashed rgba(78,168,255,.3)"></div>'
      + '<div style="position:absolute;left:50%;top:50%;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;background:#4ea8ff;box-shadow:0 0 22px rgba(78,168,255,.8);animation:xOrbit 4.5s linear infinite"></div>'
      + '<div style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);display:flex;flex-direction:column;align-items:center;gap:4px">'
      + '<div style="font-family:var(--mono);font-size:30px;font-weight:700;color:#dceaff">×500</div>'
      + '<div style="font-family:var(--display);font-size:10.5px;letter-spacing:2px;color:#6d80a0">ITERATIONS</div></div>'
      + nodes + '</div>'
      + '<div style="flex:1 1 auto;display:flex;flex-direction:column;gap:20px">'
      + '<div style="' + CAP11 + '">CATCH RATE OVER THE RUN</div>'
      + '<div style="height:230px;display:flex;align-items:flex-end;gap:7px">' + bars + '</div>'
      + '<div style="' + NOTE + '">No cliffs, no miracle jumps. Just a line that keeps going up — which is exactly why half the industry trains on PPO.</div>'
      + '</div></div>';
  };

  /* GA 1 — forty-eight of them, and one opened up to show it is only a list of numbers. */
  FIG.ga1 = function (d) {
    var crowd = d.pop.map(function (p) {
      var tom = spriteAt(cat(), 'down', p.f, 28, 52, 40);
      return '<div style="width:56px;height:56px;position:relative;overflow:hidden;border-radius:12px;border:1.5px solid rgba(61,220,132,.35);background:rgba(8,40,25,.3);opacity:' + p.o + '">'
        + (tom || '<div style="position:absolute;inset:9px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-family:var(--display);font-size:14px;color:#0b1018;background:#7e90ad">T</div>')
        + '</div>';
    }).join('');
    var genome = d.geneA.map(function (g) {
      return '<div style="flex:1 1 0;height:' + g.h + ';border-radius:4px;background:#3ddc84;transform-origin:bottom;animation:xRise .4s ' + g.delay + ' ease-out both"></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;align-items:center;gap:44px">'
      + '<div style="flex:1 1 auto;display:flex;flex-direction:column;gap:18px;animation:xUp .34s ease-out both">'
      + '<div style="' + CAP + '">GENERATION 1 · 48 TOMS, ALL RANDOM</div>'
      + '<div style="display:flex;flex-wrap:wrap;gap:11px;max-width:560px">' + crowd + '</div></div>'
      + '<div style="width:420px;flex:0 0 auto;display:flex;flex-direction:column;gap:16px;animation:xSlideR .38s .12s ease-out both">'
      + '<div style="' + CAP + '">ONE TOM, OPENED UP</div>'
      + '<div style="padding:24px;border-radius:16px;border:1px solid rgba(61,220,132,.28);background:rgba(8,40,25,.4);display:flex;flex-direction:column;gap:14px">'
      + '<div style="display:flex;align-items:flex-end;gap:5px;height:86px">' + genome + '</div>'
      + '<div style="font-family:var(--mono);font-size:12px;color:#9af0be">genome · just a list of numbers</div></div>'
      + '<div style="font-size:15px;line-height:1.5;color:#8fa4c4">Same wiring in every Tom. Only these numbers differ — and at generation 1 they are pure noise.</div>'
      + '</div></div>';
  };

  /* GA 2 — the exam. One whole life collapses into one number and that is all there is. */
  FIG.ga2 = function (d) {
    var rows = d.board.map(function (b) {
      return '<div style="display:flex;align-items:center;gap:15px;animation:xUp .3s ' + b.delay + ' ease-out both">'
        + '<div style="width:52px;font-family:var(--mono);font-size:13px;color:#6d80a0">' + b.name + '</div>'
        + '<div style="width:38px;height:38px;flex:0 0 auto;position:relative;overflow:hidden;border-radius:10px;border:1.5px solid rgba(61,220,132,.4);background:rgba(8,40,25,.3)">'
        + (spriteAt(cat(), 'down', 0, 19, 35, 30) || '') + '</div>'
        + '<div style="flex:1 1 auto;height:20px;border-radius:8px;background:rgba(255,255,255,.05);overflow:hidden">'
        + '<div style="height:100%;width:' + b.w + ';border-radius:8px;background:' + b.bg + ';transform-origin:left;animation:xBar .5s ' + b.d2 + ' ease-out both"></div></div>'
        + '<div style="width:48px;text-align:right;font-family:var(--mono);font-size:14px;font-weight:700;color:' + b.color + '">' + b.score + '</div></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;flex-direction:column;justify-content:center;gap:24px;padding:0 24px">'
      + '<div style="display:flex;justify-content:space-between;align-items:baseline;animation:xUp .3s ease-out both">'
      + '<div style="' + CAP11 + '">EVERY TOM RUNS THE LEVEL SET ONCE</div>'
      + '<div style="font-family:var(--mono);font-size:12px;color:#6d80a0">one whole life → one number</div></div>'
      + rows
      + '<div style="margin-top:6px;font-size:15px;line-height:1.5;color:#8fa4c4;animation:xUp .4s .55s ease-out both">'
      + 'Nobody asks <i>why</i> #07 scored 82. There is no bookie here, no per-step credit, no gradient. Just the number.</div></div>';
  };

  /* GA 3 — the cull. The bottom six literally fall off the screen. */
  FIG.ga3 = function (d) {
    var rows = d.cull.map(function (b) {
      return '<div style="display:flex;align-items:center;gap:15px;animation:' + b.anim + '">'
        + '<div style="width:52px;font-family:var(--mono);font-size:13px;color:#6d80a0">' + b.name + '</div>'
        + '<div style="width:38px;height:38px;flex:0 0 auto;position:relative;overflow:hidden;border-radius:10px;border:1.5px solid ' + b.ring + ';background:rgba(8,40,25,.3)">'
        + (spriteAt(cat(), 'down', 0, 19, 35, 30) || '') + '</div>'
        + '<div style="flex:1 1 auto;height:20px;border-radius:8px;background:rgba(255,255,255,.05);overflow:hidden">'
        + '<div style="height:100%;width:' + b.w + ';border-radius:8px;background:' + b.bg + '"></div></div>'
        + '<div style="width:48px;text-align:right;font-family:var(--mono);font-size:14px;font-weight:700;color:' + b.color + '">' + b.score + '</div>'
        + '<div style="width:96px;height:28px;display:flex;align-items:center;justify-content:center;border-radius:8px;font-family:var(--display);font-size:11px;letter-spacing:1.8px;background:' + b.tagBg + ';border:1px solid ' + b.tagBorder + ';color:' + b.tagColor + '">' + b.tag + '</div></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;flex-direction:column;justify-content:center;gap:22px;padding:0 24px">'
      + '<div style="' + CAP11 + ';animation:xUp .3s ease-out both">SORTED. TOP QUARTER LIVES.</div>'
      + rows
      + '<div style="margin-top:6px;padding:16px 20px;border-radius:13px;background:rgba(61,220,132,.07);border:1px solid rgba(61,220,132,.2);font-size:15px;line-height:1.5;color:#9af0be;animation:xUp .4s .5s ease-out both">'
      + 'Notice what just happened: the population got better and <b>not one Tom learned anything</b>. The graveyard did the thinking.</div></div>';
  };

  /* GA 4 — crossover, with the splice point marked in gold. */
  FIG.ga4 = function (d) {
    var kid = d.kid.map(function (g) {
      return '<div style="flex:1 1 0;height:' + g.h + ';border-radius:4px;background:' + g.bg + ';transform-origin:bottom;animation:xRise .35s ' + g.delay + ' ease-out both"></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:26px">'
      + '<div style="display:flex;gap:40px;align-items:center">'
      + '<div style="width:330px;display:flex;flex-direction:column;gap:12px;animation:xSlideL .38s ease-out both">'
      + '<div style="font-family:var(--display);font-size:10.5px;letter-spacing:2.2px;color:#3ddc84">SURVIVOR A · GOOD AT CORNERS</div>'
      + '<div style="display:flex;align-items:flex-end;gap:5px;height:80px;padding:14px;border-radius:13px;border:1px solid rgba(61,220,132,.26);background:rgba(8,40,25,.35)">'
      + geneStrip(d.geneA) + '</div></div>'
      + '<div style="font-family:var(--display);font-size:30px;color:#3ddc84;animation:xPop .4s .2s ease-out both">+</div>'
      + '<div style="width:330px;display:flex;flex-direction:column;gap:12px;animation:xSlideR .38s ease-out both">'
      + '<div style="font-family:var(--display);font-size:10.5px;letter-spacing:2.2px;color:#9af0be">SURVIVOR B · PATIENT AT THE HOLE</div>'
      + '<div style="display:flex;align-items:flex-end;gap:5px;height:80px;padding:14px;border-radius:13px;border:1px solid rgba(154,240,190,.26);background:rgba(8,40,25,.35)">'
      + geneStrip(d.geneB) + '</div></div></div>'
      + '<div style="font-family:var(--display);font-size:26px;color:#5f7392;animation:xUp .3s .3s ease-out both">↓</div>'
      + '<div style="width:520px;display:flex;flex-direction:column;gap:12px;animation:xPop .45s .38s ease-out both">'
      + '<div style="font-family:var(--display);font-size:10.5px;letter-spacing:2.2px;color:#cddcf2">THE KID · FIRST HALF FROM A, SECOND HALF FROM B</div>'
      + '<div style="display:flex;align-items:flex-end;gap:6px;height:96px;padding:16px;border-radius:14px;border:1px solid rgba(61,220,132,.4);background:rgba(8,40,25,.5);position:relative">'
      + kid
      + '<div style="position:absolute;left:50%;top:6px;bottom:6px;width:2px;background:rgba(255,209,102,.6)"></div></div>'
      + '<div style="font-size:14.5px;line-height:1.45;color:#8fa4c4;text-align:center">Sometimes he inherits both talents. Sometimes neither. The next exam will sort him out.</div>'
      + '</div></div>';
  };

  /* GA 5 — mutation. Two genes go gold, and the copy says why that is the whole engine. */
  FIG.ga5 = function (d) {
    var bars = d.mut.map(function (g) {
      return '<div style="flex:1 1 0;height:' + g.h + ';border-radius:5px;background:' + g.bg + ';box-shadow:' + g.glow + ';transform-origin:bottom;animation:xRise .4s ' + g.delay + ' ease-out both"></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;align-items:center;gap:44px">'
      + '<div style="flex:1 1 auto;display:flex;flex-direction:column;gap:16px;animation:xUp .34s ease-out both">'
      + '<div style="font-family:var(--display);font-size:10.5px;letter-spacing:2.2px;color:#5f7392">THE KID, WITH TWO RANDOM TYPOS</div>'
      + '<div style="display:flex;align-items:flex-end;gap:8px;height:150px;padding:20px;border-radius:15px;border:1px solid rgba(61,220,132,.3);background:rgba(8,40,25,.42);position:relative">' + bars + '</div>'
      + '<div style="display:flex;gap:10px;align-items:center">'
      + '<div style="width:12px;height:12px;border-radius:3px;background:#f2b544"></div>'
      + '<div style="font-family:var(--mono);font-size:12.5px;color:#f2b544">two genes jittered at random · mutation rate 0.08</div></div></div>'
      + '<div style="width:400px;flex:0 0 auto;display:flex;flex-direction:column;gap:18px;animation:xSlideR .38s .1s ease-out both">'
      + '<div style="padding:22px;border-radius:15px;border:1px solid rgba(255,209,102,.28);background:rgba(48,38,12,.34);display:flex;flex-direction:column;gap:10px">'
      + '<div style="font-family:var(--display);font-size:11px;letter-spacing:1.8px;color:#ffd166">LOOKS LIKE VANDALISM</div>'
      + '<div style="font-size:15.5px;line-height:1.5;color:#cddcf2">It is the single most important line in the algorithm.</div></div>'
      + '<div style="font-size:15.5px;line-height:1.55;color:#8fa4c4;text-wrap:pretty">Without typos, survivors just keep breeding with their own cousins. Ten generations later the whole population is one identical Tom, stuck forever. Mutation is the <b>only</b> door a genuinely new idea can walk through.</div>'
      + '</div></div>';
  };

  /* GA 6 — the generation loop, and the staircase curve it produces. */
  FIG.ga6 = function (d) {
    var loop = d.gaLoop.map(function (l) {
      return '<div style="height:54px;display:flex;align-items:center;gap:14px;padding:0 18px;border-radius:12px;border:1px solid rgba(61,220,132,.3);background:rgba(8,40,25,.34);animation:xUp .3s ' + l.delay + ' ease-out both">'
        + '<div style="font-family:var(--mono);font-size:12px;color:#3ddc84">' + l.num + '</div>'
        + '<div style="font-family:var(--display);font-size:13px;letter-spacing:1.8px;color:#9af0be;white-space:nowrap">' + l.name + '</div></div>';
    }).join('');
    var bars = d.curveGa.map(function (c) {
      return '<div style="flex:1 1 0;height:' + c.h + ';border-radius:5px 5px 0 0;background:linear-gradient(180deg,#3ddc84,rgba(61,220,132,.22));transform-origin:bottom;animation:xRise .45s ' + c.delay + ' ease-out both"></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;align-items:center;gap:44px">'
      + '<div style="width:360px;flex:0 0 auto;display:flex;flex-direction:column;gap:14px;animation:xUp .34s ease-out both">'
      + loop
      + '<div style="text-align:center;font-family:var(--display);font-size:12px;letter-spacing:2px;color:#5f7392">↻ ×200 GENERATIONS</div></div>'
      + '<div style="flex:1 1 auto;display:flex;flex-direction:column;gap:18px">'
      + '<div style="' + CAP11 + '">BEST FITNESS PER GENERATION</div>'
      + '<div style="height:250px;display:flex;align-items:flex-end;gap:6px">' + bars + '</div>'
      + '<div style="' + NOTE + '">Long flat plateaus, then somebody is born weird and useful and the whole crowd jumps. Wasteful, slow — and fearless enough to stumble into strategies a gradient would never risk.</div>'
      + '</div></div>';
  };

  /* CMA-ES 1 — centre and spread, the algorithm's entire memory. */
  FIG.cmaes1 = function () {
    return '<div style="width:100%;height:100%;display:flex;align-items:center;gap:44px">'
      + '<div style="' + CLOUD + ';animation:xPop .4s ease-out both">'
      + '<div style="' + GRID40 + '"></div>'
      + '<div style="position:absolute;left:210px;top:220px;width:250px;height:250px;margin:-125px 0 0 -125px;border-radius:50%;border:1.5px solid rgba(169,124,255,.5);background:radial-gradient(circle,rgba(169,124,255,.22),rgba(169,124,255,0) 70%);animation:xBreathe 3.4s ease-in-out infinite"></div>'
      + '<div style="position:absolute;left:210px;top:220px;width:16px;height:16px;margin:-8px 0 0 -8px;border-radius:50%;background:#a97cff;box-shadow:0 0 20px rgba(169,124,255,.9)"></div>'
      + '<div style="position:absolute;left:232px;top:190px;font-family:var(--display);font-size:11.5px;letter-spacing:1.6px;color:#e2d9ff">CENTRE · BEST GUESS</div>'
      + '<div style="position:absolute;left:210px;top:220px;width:125px;height:2px;background:linear-gradient(90deg,rgba(205,178,255,.9),rgba(205,178,255,.25))"></div>'
      + '<div style="position:absolute;left:256px;top:230px;font-family:var(--display);font-size:11.5px;letter-spacing:1.6px;color:#cdb2ff">SPREAD · HOW WIDE TO LOOK</div>'
      // The one line of this figure a viewer can misread — that the dots are just dots.
      // Spelling it out as dot = Tom costs one sprite and removes the whole ambiguity.
      + '<div style="position:absolute;left:16px;bottom:8px;display:flex;align-items:center;gap:9px">'
      + '<div style="width:11px;height:11px;flex:0 0 auto;border-radius:50%;background:#a97cff;box-shadow:0 0 12px rgba(169,124,255,.8)"></div>'
      + '<div style="font-family:var(--mono);font-size:12px;color:#8b7bb8">=</div>'
      + (spriteBox(cat(), 'down', 0, 28, 'flex:0 0 auto;margin:0 -2px') || '')
      + '<div style="font-family:var(--mono);font-size:11px;color:#5f7392">every point on this map is one complete Tom</div></div></div>'
      + '<div style="flex:1 1 auto;min-width:0;display:flex;flex-direction:column;gap:20px;animation:xSlideR .38s .1s ease-out both">'
      + '<div style="padding:22px;border-radius:15px;border:1px solid rgba(169,124,255,.28);background:rgba(38,19,77,.42);display:flex;flex-direction:column;gap:9px">'
      + '<div style="font-family:var(--display);font-size:11px;letter-spacing:1.8px;color:#cdb2ff">THE ENTIRE MEMORY OF THE ALGORITHM</div>'
      + '<div style="font-size:16px;line-height:1.5;color:#e2d9ff">1 · where we think the answer is<br>2 · how confused we still are</div></div>'
      + '<div style="font-size:15.5px;line-height:1.55;color:#8fa4c4;text-wrap:pretty">No population to feed, no gradient to compute. Just a cloud that knows where to look next.</div>'
      + '</div></div>';
  };

  /* CMA-ES 2 — sixteen samples drawn out of the cloud, four of them winners. */
  FIG.cmaes2 = function (d) {
    var dots = d.samples.map(function (s) {
      return '<div style="position:absolute;left:' + s.left + ';top:' + s.top + ';width:' + s.size + ';height:' + s.size
        + ';border-radius:50%;background:' + s.bg + ';box-shadow:' + s.glow + ';animation:xPop .34s ' + s.delay + ' ease-out both"></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;align-items:center;gap:44px">'
      + '<div style="' + CLOUD + ';animation:xUp .34s ease-out both">'
      + '<div style="' + GRID40 + '"></div>'
      + '<div style="position:absolute;left:210px;top:220px;width:250px;height:250px;margin:-125px 0 0 -125px;border-radius:50%;border:1.5px solid rgba(169,124,255,.4);background:radial-gradient(circle,rgba(169,124,255,.14),rgba(169,124,255,0) 70%)"></div>'
      + dots
      + '<div style="position:absolute;left:210px;top:220px;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;background:#a97cff;box-shadow:0 0 18px rgba(169,124,255,.9)"></div></div>'
      + '<div style="flex:1 1 auto;min-width:0;display:flex;flex-direction:column;gap:20px;animation:xSlideR .38s .12s ease-out both">'
      + '<div style="display:flex;gap:14px;align-items:center">'
      + '<div style="width:14px;height:14px;border-radius:50%;background:#c9a6ff;box-shadow:0 0 14px rgba(201,166,255,.9)"></div>'
      + '<div style="font-size:15.5px;color:#cddcf2">the four that scored best</div></div>'
      + '<div style="display:flex;gap:14px;align-items:center">'
      + '<div style="width:11px;height:11px;border-radius:50%;background:rgba(169,124,255,.42)"></div>'
      + '<div style="font-size:15.5px;color:#8fa4c4">the twelve that didn’t</div></div>'
      + '<div style="height:1px;background:rgba(140,170,210,.14)"></div>'
      + '<div style="font-size:15.5px;line-height:1.55;color:#8fa4c4;text-wrap:pretty">Same brutal one-number exam as the genetic school. The difference is <b>where the candidates come from</b> — not random guessing, but drawn out of the cloud.</div>'
      + '</div></div>';
  };

  /* CMA-ES 3 — the centre glides onto the winners' centre of gravity. */
  FIG.cmaes3 = function (d) {
    var dots = d.samples.map(function (s) {
      return '<div style="position:absolute;left:' + s.left + ';top:' + s.top + ';width:' + s.size + ';height:' + s.size
        + ';border-radius:50%;background:' + s.bg + ';box-shadow:' + s.glow + '"></div>';
    }).join('');
    var weights = d.weights.map(function (w) {
      return '<div style="display:flex;align-items:center;gap:14px">'
        + '<div style="width:76px;font-family:var(--display);font-size:12px;letter-spacing:1.4px;color:#cdb2ff">' + w.name + '</div>'
        + '<div style="flex:1 1 auto;height:16px;border-radius:8px;background:rgba(255,255,255,.05);overflow:hidden">'
        + '<div style="height:100%;width:' + w.w + ';border-radius:8px;background:#a97cff;transform-origin:left;animation:xBar .5s ' + w.delay + ' ease-out both"></div></div>'
        + '<div style="width:48px;text-align:right;font-family:var(--mono);font-size:13px;font-weight:700;color:#e2d9ff">' + w.v + '</div></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;align-items:center;gap:44px">'
      + '<div style="' + CLOUD + ';animation:xUp .34s ease-out both">'
      + '<div style="' + GRID40 + '"></div>'
      + '<div style="position:absolute;left:210px;top:220px;width:250px;height:250px;margin:-125px 0 0 -125px;border-radius:50%;border:1.5px dashed rgba(169,124,255,.24)"></div>'
      + dots
      + '<div style="position:absolute;left:210px;top:220px;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;background:rgba(169,124,255,.35)"></div>'
      + '<div style="position:absolute;left:210px;top:220px;width:20px;height:20px;margin:-10px 0 0 -10px;border-radius:50%;background:#a97cff;box-shadow:0 0 26px rgba(169,124,255,1);animation:xGlide 1.3s .25s cubic-bezier(.4,.9,.3,1) both"></div>'
      + '<div style="position:absolute;left:214px;top:216px;width:64px;height:2px;transform-origin:left;transform:rotate(-36deg);background:linear-gradient(90deg,rgba(205,178,255,.85),rgba(205,178,255,.3));animation:xBar .6s .25s ease-out both"></div>'
      + '<div style="position:absolute;left:16px;bottom:12px;font-family:var(--mono);font-size:11px;color:#5f7392">new centre = weighted average of the winners</div></div>'
      + '<div style="flex:1 1 auto;min-width:0;display:flex;flex-direction:column;gap:18px;animation:xSlideR .38s .12s ease-out both">'
      + '<div style="font-family:var(--display);font-size:11px;letter-spacing:2.2px;color:#5f7392">HOW MUCH EACH WINNER COUNTS</div>'
      + weights
      + '<div style="font-size:15.5px;line-height:1.55;color:#8fa4c4;text-wrap:pretty">The very best dot pulls hardest. The losers aren’t deleted — they simply pull with a weight of zero.</div>'
      + '</div></div>';
  };

  /* CMA-ES 4 — the covariance step: the circle stretches into a valley-shaped ellipse. */
  FIG.cmaes4 = function (d) {
    var ridge = d.ridge.map(function (r) {
      return '<div style="position:absolute;left:' + r.left + ';top:' + r.top + ';width:13px;height:13px;border-radius:50%;background:#c9a6ff;box-shadow:0 0 16px rgba(201,166,255,.9)"></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;align-items:center;gap:44px">'
      + '<div style="' + CLOUD + ';animation:xUp .34s ease-out both">'
      + '<div style="' + GRID40 + '"></div>'
      + '<div style="position:absolute;left:260px;top:210px;width:250px;height:250px;margin:-125px 0 0 -125px;border-radius:50%;border:1.5px solid rgba(169,124,255,.55);background:radial-gradient(circle,rgba(169,124,255,.2),rgba(169,124,255,0) 70%);animation:xEllipse 1.6s .2s cubic-bezier(.4,.9,.3,1) both"></div>'
      + ridge
      + '<div style="position:absolute;left:260px;top:210px;width:15px;height:15px;margin:-7px 0 0 -7px;border-radius:50%;background:#a97cff;box-shadow:0 0 22px rgba(169,124,255,1)"></div>'
      + '<div style="position:absolute;left:16px;bottom:12px;font-family:var(--mono);font-size:11px;color:#5f7392">stretch along what mattered, squeeze what didn’t</div></div>'
      + '<div style="flex:1 1 auto;min-width:0;display:flex;flex-direction:column;gap:20px;animation:xSlideR .38s .12s ease-out both">'
      + '<div style="padding:22px;border-radius:15px;border:1px solid rgba(169,124,255,.3);background:rgba(38,19,77,.42);display:flex;flex-direction:column;gap:9px">'
      + '<div style="font-family:var(--display);font-size:11px;letter-spacing:1.8px;color:#cdb2ff">THIS IS THE CMA IN CMA-ES</div>'
      + '<div style="font-size:16px;line-height:1.5;color:#e2d9ff">Covariance Matrix Adaptation — a fancy name for “the cloud learns which directions are worth trying”.</div></div>'
      + '<div style="font-size:15.5px;line-height:1.55;color:#8fa4c4;text-wrap:pretty">After a few rounds the cloud is a long thin ellipse pointing straight down the useful valley. It hasn’t just learned <b>where</b> the answer is — it has learned the <b>shape of the problem</b>.</div>'
      + '</div></div>';
  };

  /* CMA-ES 5 — sigma control, read off the algorithm's own momentum. */
  FIG.cmaes5 = function () {
    return '<div style="width:100%;height:100%;display:flex;flex-direction:column;justify-content:center;gap:26px;padding:0 20px">'
      + '<div style="' + CAP11 + ';animation:xUp .3s ease-out both">ONE MORE DIAL: HOW BIG A STRIDE TO TAKE</div>'
      + '<div style="display:flex;gap:24px">'
      + '<div style="flex:1 1 0;padding:26px;border-radius:16px;border:1px solid rgba(169,124,255,.3);background:rgba(38,19,77,.34);display:flex;flex-direction:column;gap:18px;animation:xSlideL .38s ease-out both">'
      + '<div style="font-family:var(--display);font-size:12px;letter-spacing:1.8px;color:#cdb2ff">SAME DIRECTION, ROUND AFTER ROUND</div>'
      + '<div style="height:70px;display:flex;align-items:center;gap:12px">'
      + '<div style="font-size:26px;color:#a97cff">→</div><div style="font-size:26px;color:#a97cff">→</div>'
      + '<div style="font-size:26px;color:#a97cff">→</div><div style="font-size:26px;color:#a97cff">→</div>'
      + '<div style="flex:1 1 auto"></div>'
      + '<div style="width:74px;height:74px;border-radius:50%;background:radial-gradient(circle,rgba(169,124,255,.4),rgba(169,124,255,0) 70%);border:1.5px solid rgba(169,124,255,.6);animation:xPop .5s .25s ease-out both"></div></div>'
      + '<div style="font-size:15.5px;line-height:1.5;color:#cddcf2">Clearly a long slope. <b>Grow σ</b> — sprint.</div></div>'
      + '<div style="flex:1 1 0;padding:26px;border-radius:16px;border:1px solid rgba(130,160,200,.2);background:rgba(255,255,255,.02);display:flex;flex-direction:column;gap:18px;animation:xSlideR .38s .1s ease-out both">'
      + '<div style="font-family:var(--display);font-size:12px;letter-spacing:1.8px;color:#9db1cd">ZIG-ZAGGING BACK AND FORTH</div>'
      + '<div style="height:70px;display:flex;align-items:center;gap:12px">'
      + '<div style="font-size:26px;color:#7d90ad">→</div><div style="font-size:26px;color:#7d90ad">←</div>'
      + '<div style="font-size:26px;color:#7d90ad">→</div><div style="font-size:26px;color:#7d90ad">←</div>'
      + '<div style="flex:1 1 auto"></div>'
      + '<div style="width:30px;height:30px;border-radius:50%;background:radial-gradient(circle,rgba(169,124,255,.4),rgba(169,124,255,0) 70%);border:1.5px solid rgba(169,124,255,.6);animation:xPop .5s .25s ease-out both"></div></div>'
      + '<div style="font-size:15.5px;line-height:1.5;color:#cddcf2">We’re circling the answer. <b>Shrink σ</b> — tiptoe.</div></div></div>'
      + '<div style="padding:16px 20px;border-radius:13px;background:rgba(169,124,255,.07);border:1px solid rgba(169,124,255,.2);font-size:15.5px;line-height:1.5;color:#cdb2ff;animation:xUp .4s .3s ease-out both">'
      + 'The cloud checks its own momentum and decides how brave to be. No human tunes this — which is why CMA-ES is famous for working out of the box.</div></div>';
  };

  /* CMA-ES 6 — convergence, and the honest caveat about where it loses to a GA. */
  FIG.cmaes6 = function (d) {
    var rings = d.converge.map(function (c) {
      return '<div style="position:absolute;left:50%;top:50%;width:' + c.s + ';height:' + c.s + ';margin:' + c.m
        + ';opacity:' + c.o + ';border-radius:50%;border:1.5px solid rgba(169,124,255,.4);animation:xPop .4s ' + c.delay + ' ease-out both"></div>';
    }).join('');
    return '<div style="width:100%;height:100%;display:flex;align-items:center;gap:44px">'
      + '<div style="width:430px;height:400px;flex:0 0 auto;position:relative;animation:xUp .34s ease-out both">'
      + rings
      + '<div style="position:absolute;left:50%;top:50%;width:20px;height:20px;margin:-10px 0 0 -10px;border-radius:50%;background:#a97cff;box-shadow:0 0 28px rgba(169,124,255,1);animation:xPop .4s .7s ease-out both"></div>'
      + '<div style="position:absolute;left:50%;top:50%;width:60px;height:60px;margin:-30px 0 0 -30px;border-radius:50%;border:2px solid rgba(169,124,255,.6);animation:xRing 1.8s .8s ease-out infinite"></div></div>'
      + '<div style="flex:1 1 auto;min-width:0;display:flex;flex-direction:column;gap:18px;animation:xSlideR .38s .1s ease-out both">'
      + '<div style="display:flex;gap:16px">'
      + '<div style="flex:1 1 0;padding:18px;border-radius:14px;border:1px solid rgba(169,124,255,.26);background:rgba(38,19,77,.34);display:flex;flex-direction:column;gap:6px">'
      + '<div style="font-family:var(--display);font-size:10.5px;letter-spacing:1.8px;color:#cdb2ff">STRENGTH</div>'
      + '<div style="font-size:15px;line-height:1.45;color:#e2d9ff">Very few runs wasted. No gradients. Almost nothing to tune.</div></div>'
      + '<div style="flex:1 1 0;padding:18px;border-radius:14px;border:1px solid rgba(255,209,102,.24);background:rgba(48,38,12,.3);display:flex;flex-direction:column;gap:6px">'
      + '<div style="font-family:var(--display);font-size:10.5px;letter-spacing:1.8px;color:#ffd166">THE CATCH</div>'
      + '<div style="font-size:15px;line-height:1.45;color:#f0e2c4">It wants a fairly smooth landscape. On a spiky one, 48 brawling Toms can still win.</div></div></div>'
      + '<div style="height:1px;background:rgba(140,170,210,.14)"></div>'
      + '<div style="font-size:15.5px;line-height:1.55;color:#8fa4c4;text-wrap:pretty">Round after round the cloud tightens until every sample is basically the same Tom. That’s convergence — and the last centre is your trained agent.</div>'
      + '</div></div>';
  };

  /* ---------------- the overlay ---------------- */

  function count(key) { return (STEPS[key] || STEPS.ppo).length; }

  /* `v` is a Reveal view, never a raw ALGOS entry — the caller has already refused to
     open this for a sealed school, and the accent comes from the view either way.
     `animate` fades the whole panel in and belongs to the opening only: on a step the
     chrome is identical, so re-creating it silently leaves the new diagram to play alone. */
  function overlay(v, step, animate) {
    var list = STEPS[v.key] || STEPS.ppo;
    var i = Math.max(1, Math.min(list.length, step || 1));
    var cur = list[i - 1];
    var last = i === list.length;
    var d = data();
    var border = P.rgba(v.color, .34);

    var dots = list.map(function (_, n) {
      return '<div data-xstep="' + (n + 1) + '" style="cursor:pointer;width:38px;height:6px;border-radius:3px;background:'
        + (n + 1 === i ? v.color : (n + 1 < i ? P.rgba(v.color, .4) : 'rgba(130,160,200,.2)')) + '"></div>';
    }).join('');

    var body = cur.b.map(function (t) {
      return '<div style="font-size:17.5px;line-height:1.55;color:#b3c5df;text-wrap:pretty">' + esc(t) + '</div>';
    }).join('');

    var fig = FIG[v.key + i];
    var figure = fig ? fig(d) : '';

    return '<div style="position:absolute;inset:0;z-index:80;padding:26px 30px;display:flex;flex-direction:column;gap:20px;background:radial-gradient(120% 90% at 50% -10%,#0b1424 0%,#04070d 55%,#02040a 100%);font-family:var(--body)'
      + (animate ? ';animation:xIn .18s ease-out' : '') + '">'
      + '<div class="grid-lines" style="opacity:.26"></div>'

      + '<div style="flex:0 0 auto;height:64px;display:flex;align-items:center;gap:18px;position:relative">'
      + '<div style="width:48px;height:48px;flex:0 0 auto;display:flex;align-items:center;justify-content:center;border-radius:13px;background:' + v.deep + ';border:1px solid ' + border + '">'
      + '<div style="width:28px;height:28px">' + P.emblem(v.emblem, v.color) + '</div></div>'
      + '<div style="display:flex;flex-direction:column;gap:3px">'
      + '<div style="font-family:var(--display);font-weight:600;font-size:27px;letter-spacing:2.4px;line-height:1;color:' + v.color + '">HOW ' + esc(v.short) + ' WORKS</div>'
      + '<div style="font-family:var(--mono);font-size:11px;color:#6d80a0">' + esc(v.full) + ' · six steps, no maths</div></div>'
      + '<div style="flex:1 1 auto"></div>'
      + '<div style="display:flex;gap:7px;flex:0 0 auto">' + dots + '</div>'
      + '<div data-act="x-close" style="cursor:pointer;flex:0 0 auto;height:38px;padding:0 17px;display:flex;align-items:center;gap:9px;border-radius:10px;border:1px solid rgba(130,160,200,.24);background:rgba(255,255,255,.03);font-family:var(--display);font-size:11.5px;letter-spacing:1.6px;color:#9db1cd">✕ BACK TO TRAINING</div>'
      + '</div>'

      + '<div style="flex:1 1 auto;min-height:0;display:flex;gap:24px;position:relative">'
      + '<div style="width:552px;flex:0 0 auto;display:flex;flex-direction:column;gap:20px;padding:34px 34px 30px;border-radius:20px;border:1px solid rgba(130,160,200,.16);background:linear-gradient(180deg,rgba(16,24,38,.72),rgba(8,12,20,.6))">'
      + '<div style="flex:0 0 auto;display:flex;align-items:center;gap:12px">'
      + '<div style="font-family:var(--display);font-size:11.5px;letter-spacing:2.6px;color:' + v.color + '">' + esc(cur.k) + '</div>'
      + '<div style="flex:1 1 auto;height:1px;background:rgba(140,170,210,.16)"></div></div>'
      + '<div style="flex:0 0 auto;font-family:var(--display);font-weight:600;font-size:40px;line-height:1.1;letter-spacing:.5px;color:#f2f7ff;text-wrap:pretty">' + esc(cur.t) + '</div>'
      + '<div style="flex:1 1 auto;min-height:0;display:flex;flex-direction:column;gap:17px">' + body + '</div>'
      + '<div style="flex:0 0 auto;display:flex;gap:13px;align-items:flex-start;padding:16px 18px;border-radius:13px;background:' + P.rgba(v.color, .08) + ';border:1px solid ' + border + '">'
      + '<div style="width:6px;align-self:stretch;border-radius:3px;background:' + v.color + ';flex:0 0 auto"></div>'
      + '<div style="display:flex;flex-direction:column;gap:5px">'
      + '<div style="font-family:var(--display);font-size:9.5px;letter-spacing:2px;color:#7d90ad">IN ONE LINE</div>'
      + '<div style="font-size:17px;line-height:1.4;font-weight:600;color:' + v.light + ';text-wrap:pretty">' + esc(cur.take) + '</div>'
      + '</div></div></div>'

      + '<div data-act="x-next" style="flex:1 1 auto;min-width:0;cursor:pointer;position:relative;border-radius:20px;overflow:hidden;border:1px solid rgba(130,160,200,.14);background:linear-gradient(180deg,rgba(10,15,25,.9),rgba(6,9,16,.9))">'
      // The figures are authored on a fixed 1190x804 board and scaled into the pane, so a
      // diagram is laid out once and never has to reflow for the space it lands in.
      + '<div style="position:absolute;left:0;top:0;width:1190px;height:804px;transform:scale(1.079);transform-origin:top left;padding:28px 32px">'
      + figure + '</div>'
      + '<div style="position:absolute;right:18px;bottom:14px;font-family:var(--mono);font-size:11px;color:#3f506b">click anywhere to continue →</div>'
      + '</div></div>'

      + '<div style="flex:0 0 auto;height:56px;display:flex;align-items:center;gap:14px;position:relative">'
      + '<div data-act="x-prev" style="cursor:pointer;height:48px;padding:0 24px;display:flex;align-items:center;border-radius:12px;border:1px solid rgba(130,160,200,.24);background:rgba(255,255,255,.03);font-family:var(--display);font-size:12.5px;letter-spacing:1.8px;color:'
      + (i === 1 ? '#3f506b' : '#9db1cd') + '">← PREVIOUS</div>'
      + '<div style="font-family:var(--mono);font-size:13px;color:#6d80a0">'
      + String(i).padStart(2, '0') + ' / ' + String(list.length).padStart(2, '0') + '</div>'
      + '<div style="flex:1 1 auto;text-align:center;font-size:13.5px;color:#5f7392">arrow keys or space to step through · esc to go back</div>'
      + '<div data-act="x-next" style="cursor:pointer;height:48px;padding:0 30px;display:flex;align-items:center;border-radius:12px;background:'
      + P.rgba(v.color, last ? .26 : .16) + ';border:1px solid ' + border + ';font-family:var(--display);font-size:12.5px;letter-spacing:1.8px;color:' + v.light + '">'
      + (last ? 'BACK TO TRAINING ✓' : 'NEXT →') + '</div>'
      + '</div></div>';
  }

  global.Explain = { STEPS: STEPS, count: count, overlay: overlay };
})(window);
