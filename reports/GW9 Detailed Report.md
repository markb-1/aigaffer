# GW9 Detailed Report

Deadline: Sat 31 Oct 2026 11:00 UTC · Last updated: scout run Wed 28 Oct 2026 23:35 UTC

## What changed

**scout — Wed 28 Oct 2026 23:35 UTC**

- Chip: triple_captain (was: none)
- Calafiori: xMins 25 → 15
- Rice: xMins 85.8 → 70
- João Pedro: xMins 10 → 0
- Thomas: xMins 40 → 35
- Rice: news 'Unspecified injury - 75% chance of playing' (was none)
- João Pedro: news 'Thigh injury - Unknown return date' (was 'Unspecified injury - 75% chance of playing')

**early — Mon 26 Oct 2026 18:35 UTC**

- First look

## The plan

**WHAT I DID**

One free transfer: Calafiori out, Gabriel in. Triple Captain on Haaland, with Mbeumo as vice. No hit.

**WHAT I LEARNED**

- Calafiori (Arsenal): Arteta said he missed Tuesday's cup tie at Fleetwood. He hurt his hip or side after coming off the bench against Everton. Football.london report "genuine concern" that he misses Liverpool away on Sunday, and he wasn't starting before the knock anyway. I cut his minutes from 56 to 15.
- Rice (Arsenal): also missed the cup tie, with a minor problem. He trained on Monday, and the Standard and football.london expect him to be fit for Anfield. Minutes 64 → 70.
- João Pedro (Chelsea): Alonso confirmed he is out for "a few weeks" with a thigh injury. Minutes stay at 0. My guess is he's back around GW12.
- Thomas (Coventry): went off just before the hour against Fulham. Lampard didn't know what the injury was, and one predicted line-up has Mfuni in his place. Minutes 63 → 35, so he's likely benched behind Vuskovic.
- Haaland (Man City): rested, not injured. He was one of five City players given three days off and skips the midweek cup tie.
- Lewis-Potter was only a sub in Brentford's cup tie, which looks like rest.
- I also carried four earlier findings into the projection that it didn't have yet: Gabriel 88 minutes, Tavernier 85, Guéhi 72, Haaland 85.
- What I couldn't check: the search tool ran out before Man Utd (Mbeumo), Hull (Tzolakis), Everton, Bournemouth and Leeds, and the Leeds result I did get was an old article. Mbeumo, Tzolakis, Muharemović and Calvert-Lewin are going in on the model's numbers. Arteta's Friday press conference could still change things on Rice and Calafiori.

**WHY THIS PLAN**

- Selling Calafiori isn't a hunch. He's injured, wasn't starting before that, and the model had him on 56 minutes. Gabriel is a nailed Arsenal starter.
- After my changes the solver rates the move 1.9 points better than rolling the transfer (plan 6). Rolling would only make the same move next week, and in the meantime Calafiori's minutes would be wasted.
- Plan 5 also swaps Thomas for Konsa, at a 4-point hit. It comes out only 0.3 ahead of rolling, which is within the noise, so it isn't worth paying 4 points for. I'd rather make the Thomas move with a free transfer next week, once we know what his injury is.
- Plan 7 takes two hits and comes out below rolling.
- Gabriel is being bought by a lot of managers (about 164k net transfers in), so his price could rise before the deadline. That's timing only, not a reason for the move.
- The Triple Captain reasoning is in the chip justification: the minutes are confirmed, but the fixture (away at Brighton) is only average, and the edge over saving it for GW15 is small.

Expected points over the window: 246.28 xP; net 246.84 (after hits and charges, plus the credit for free transfers kept); 0 hits; bank £3.7m.

## Transfers

### Calafiori → Gabriel

| | Player | Price | GW9 | GW10 | GW11 | GW12 | GW13 | GW14 | Total | Weighted |
|---|---|---|---|---|---|---|---|---|---|---|
| Out | Calafiori (DEF, ARS) | £6.0m | liv(a) 0.81 | HUL(h) 3.62 | new(a) 3.25 | MCI(h) 3.04 | bre(a) 3.17 | tot(a) 3.45 | 17.34 | 11.28 |
| In | Gabriel (DEF, ARS) | £7.9m | liv(a) 4.82 | HUL(h) 5.85 | new(a) 5.26 | MCI(h) 4.95 | bre(a) 5.15 | tot(a) 5.57 | 31.60 | 21.79 |

Weighted counts later gameweeks for less, as the planner does; it is the total decisions are made on.

Gabriel over Calafiori: +10.51 weighted (+14.26 plain) over the window. Calafiori has a hip injury, missed the Fleetwood cup tie and was not starting before it, so I cut him to 15 minutes. He is worth 0.81 at Liverpool. Gabriel is a nailed Arsenal starter on 88 minutes and is worth 4.82 there, then 5.85 against Hull City. Over the window that is 31.60 against 17.34, or +10.51 weighted.

## Your squad

| | Player | Price | GW9 | GW10 | GW11 | GW12 | GW13 | GW14 | Total | Weighted |
|---|---|---|---|---|---|---|---|---|---|---|
| XI | Tzolakis (GKP, HUL) | £4.7m | IPS(h) 4.83 | ars(a) 3.87 | BHA(h) 4.19 | cry(a) 4.61 | NFO(h) 4.77 | bou(a) 4.13 | 26.40 | 18.29 |
| XI | Gabriel (DEF, ARS) | £7.9m | liv(a) 4.82 | HUL(h) 5.85 | new(a) 5.26 | MCI(h) 4.95 | bre(a) 5.15 | tot(a) 5.57 | 31.60 | 21.79 |
| XI | Muharemović (DEF, LEE) | £5.1m | bou(a) 3.83 | TOT(h) 4.57 | che(a) 3.71 | COV(h) 4.92 | mci(a) 3.22 | IPS(h) 4.75 | 25.00 | 17.21 |
| XI | Guéhi (DEF, MCI) | £6.0m | BHA(h) 3.71 | nfo(a) 4.22 | FUL(h) 4.67 | ars(a) 3.58 | LEE(h) 4.54 | bre(a) 3.89 | 24.61 | 16.96 |
| XI | Tavernier (MID, BOU) | £6.4m | LEE(h) 5.34 | ips(a) 5.54 | NFO(h) 5.03 | ful(a) 4.83 | BHA(h) 4.92 | HUL(h) 4.97 | 30.63 | 21.41 |
| XI | Mbeumo (MID, MUN) | £7.9m | che(a) 4.92 | AVL(h) 5.20 | liv(a) 4.69 | BRE(h) 5.23 | new(a) 5.02 | COV(h) 5.87 | 30.93 | 21.17 |
| XI | Groß (MID, BHA) | £6.2m | mci(a) 4.11 | BRE(h) 4.64 | hul(a) 4.38 | NEW(h) 4.75 | bou(a) 4.42 | nfo(a) 4.41 | 26.71 | 18.40 |
| XI | Rice (MID, ARS) | £7.4m | liv(a) 3.63 | HUL(h) 3.61 | new(a) 3.54 | MCI(h) 3.42 | bre(a) 3.48 | tot(a) 3.60 | 21.28 | 14.77 |
| XI | Lewis-Potter (MID, BRE) | £5.5m | NFO(h) 3.50 | bha(a) 3.25 | EVE(h) 3.49 | mun(a) 3.35 | ARS(h) 3.17 | MCI(h) 3.25 | 20.01 | 13.94 |
| XI | Haaland (FWD, MCI) | £15.8m | BHA(h) 5.54 | nfo(a) 5.36 | FUL(h) 5.86 | ars(a) 4.95 | LEE(h) 5.69 | bre(a) 5.49 | 32.89 | 22.77 |
| XI | Calvert-Lewin (FWD, LEE) | £6.1m | bou(a) 4.49 | TOT(h) 4.97 | che(a) 4.55 | COV(h) 5.18 | mci(a) 4.16 | IPS(h) 5.26 | 28.61 | 19.69 |
| Bench 1 | Verbruggen (GKP, BHA) | £4.6m | mci(a) 2.81 | BRE(h) 3.58 | hul(a) 3.71 | NEW(h) 3.65 | bou(a) 3.33 | nfo(a) 3.68 | 20.76 | 14.15 |
| Bench 2 | Vuskovic (DEF, BHA) | £5.2m | mci(a) 3.34 | BRE(h) 4.32 | hul(a) 4.28 | NEW(h) 4.43 | bou(a) 3.97 | nfo(a) 4.28 | 24.62 | 16.79 |
| Bench 3 | Thomas (DEF, COV) | £4.0m | SUN(h) 2.03 | eve(a) 3.19 | CRY(h) 3.52 | lee(a) 3.07 | IPS(h) 3.53 | mun(a) 2.88 | 18.22 | 12.29 |
| Bench 4 | João Pedro (FWD, CHE) | £7.8m | MUN(h) 0.00 | sun(a) 0.00 | LEE(h) 0.00 | nfo(a) 0.00 | CRY(h) 0.00 | LIV(h) 0.00 | 0.00 | 0.00 |

Weighted counts later gameweeks for less, as the planner does; it is the total decisions are made on.

## Captain and vice

- **Captain: Haaland** (32.89 over the window) — I gave Haaland the armband on his total over the window. His 32.89 beats Tavernier's 30.63 and Calvert-Lewin's 28.61. The game that settled it was GW9 against Brighton, where his 5.54 is ahead of Tavernier's 5.34 against Leeds and Calvert-Lewin's 4.49 at Bournemouth. He was rested midweek, not injured, so he starts fresh. Mbeumo is vice, on 30.93 over the window and 4.92 at Chelsea.
- **Vice: Mbeumo** (30.93 over the window)
- Rivals (by total xP over the window): Tavernier (30.63), Calvert-Lewin (28.61)

## Transfers skipped

| Player | Instead of | Gap (weighted) | Price | Why not | Note |
|---|---|---|---|---|---|
| Trafford | Verbruggen | +4.75 | +0.5m | the plan does better without him | He is +4.75 weighted over Verbruggen, but my one free transfer had to fix the injured Calafiori, and the plan does better without a hit for a goalkeeper. |
| Raya | Verbruggen | +4.69 | +1.7m | the plan does better without him | Raya is +4.69 weighted over Verbruggen, but he costs 1.7m more, and the free transfer went on Calafiori. The plan does better without him. |
| Suzuki | Verbruggen | +4.37 | +0.5m | the plan does better without him | Suzuki is +4.37 weighted over Verbruggen, the smallest goalkeeper gain of the three, and it isn't worth a hit. The plan does better without him. |
| Tarkowski | Calafiori | +8.70 | +0.5m | the plan does better without him | Tarkowski is +8.70 weighted over Calafiori, behind Gabriel's +10.51. He only came in as part of the two-hit plan 7, which nets 243.41. |
| Konsa | Calafiori | +8.52 | -1.0m | the plan does better without him | Konsa is +8.52 weighted over Calafiori, behind Gabriel's +10.51. As a Thomas replacement in plan 5, he comes out only 0.3 ahead of rolling, which doesn't justify a hit. |
| Hill | Calafiori | +7.82 | -0.2m | the plan does better without him | Hill is +7.82 weighted over Calafiori, the weakest of the defensive options. Gabriel's +10.51 was the clear pick. |
| Dewsbury-Hall | Lewis-Potter | +4.66 | +1.1m | the plan does better without him | Dewsbury-Hall is +4.66 weighted over Lewis-Potter, but he only fitted in the hit plans 2 and 3, and both net less than plan 4. |
| Ødegaard | Lewis-Potter | +4.40 | +1.2m | the plan does better without him | Ødegaard is +4.40 weighted over Lewis-Potter and costs 1.2m more. That gain doesn't cover a 4-point hit, so the plan does better without him. |
| Palmer | Lewis-Potter | +3.91 | +4.3m | unaffordable | Palmer is only +3.91 weighted over Lewis-Potter and needs 4.3m more, which I can't afford. |
| Barry | João Pedro | +17.61 | -2.0m | would sit on the bench | Barry is +17.61 weighted over the injured João Pedro, but he would sit on my bench, so those points wouldn't count. |
| N.Jackson | João Pedro | +17.45 | -0.8m | would sit on the bench | Jackson is +17.45 weighted over João Pedro, but he would sit on the bench, so the gain wouldn't count. |
| Evanilson | João Pedro | +17.21 | -1.4m | would sit on the bench | Evanilson is +17.21 weighted over João Pedro, but he would be a benched forward, so the gain wouldn't count. |

## The options weighed

| Plan | In | Out | Hits | xP | Net | |
|---|---|---|---|---|---|---|
| 0 | Gabriel | Calafiori | 0 | 246.90 | 247.46 |  |
| 1 | none | none | 0 | 245.37 | 245.92 |  |
| 2 | Gabriel, Dewsbury-Hall | Calafiori, Rice | 1 | 248.68 | 245.73 |  |
| 3 | Gabriel, Tarkowski, Dewsbury-Hall | Calafiori, Rice, Vuskovic | 2 | 250.52 | 243.98 |  |
| 4 | Gabriel | Calafiori | 0 | 246.28 | 246.84 | chosen, re-solved |
| 5 | Gabriel, Konsa | Calafiori, Thomas | 1 | 248.21 | 245.26 | re-solved |
| 6 | none | none | 0 | 244.37 | 244.93 | re-solved |
| 7 | Gabriel, Tarkowski, Dewsbury-Hall | Calafiori, Rice, Thomas | 2 | 249.94 | 243.41 | re-solved |

Plan 4 nets 246.84. Rolling the transfer (plan 6) nets 244.93, so the free transfer is worth about 1.9 points, and rolling would only make the same move next week. Plan 5 adds Thomas to Konsa for a 4-point hit and nets 245.26. That beats rolling by just 0.3, which is not worth a hit, so the Thomas move waits for a free transfer once his injury is clear. Plan 7 takes two hits and nets only 243.41.

## Chips

Chip played: Triple Captain.

I played Triple Captain on Haaland because it adds +5.69 this week, which clears its bar of 4.2. Its best later week is GW15 at +5.1, so holding it back gains nothing. Haaland's minutes are confirmed, since Maresca said he was rested, not injured. The weak point is the Brighton fixture, and that is why the margin is small. Free Hit stays saved for GW17 (+6.5), and its bar this week is 15.1. Wildcard comes out at -16.16.

Points each chip would add this week:
- Bench Boost: +9.16
- Free Hit: +7.06
- Triple Captain: +5.69
- Wildcard (over the window): -16.16

Calendar:
- Triple Captain (GW1–19): played this week (bar 4.2); its best week past the window would have been GW15 (+5.1)
- Wildcard (GW2–19): this week's bar 21.9, falling to 0 as GW19 nears
- Free Hit (GW2–19): saved for GW17, worth +6.5 there; this week's bar 15.1
- The first set expires after GW19.
