# MoQSplat reproduction and external contribution validation — 2026-11-01

User authority: validate Shao-Yang Hung's PR with new GPU-server experiments,
fix code as needed, make reproduction complete, inspect forks and reconstruct
missing transport if it cannot be recovered. No messages to authors authorized.

Scope: MoQSplat, upstream main 15e02a669456d767abf5d68f9fb1f909f260dde0;
external PR2 head 1953934487ebd8a03cd18c257695f0ecdae95aee. Preserve
uncommitted agent-configuration changes in /home/itec/emanuele/MoQSplat.
Work in isolated /tmp/moqsplat-repro-code-20261101, codex/reproduction-validation.

Decision rule: real Bicycle source and real camera trace must produce finite,
nonempty, nonblack CUDA-rendered frames through an actual network path;
camera/SH/subscription regression tests must pass. QUIC claims additionally
require native MoQ transport negotiation, independently scheduled layer streams,
relay forwarding, propagation of priority updates, complete payload identity,
and deduplication/retry/cache behavior. Do not infer these from offline VMAF.

Hypothesis: PR identifies genuine defects but needs corrections. Competing
explanations: camera-coordinate mismatch, incorrect SH layout, premature
subscription advancement and incomplete scene visibility cause apparent fixes.

Budget chosen for this bounded validation: at most 3600 GPU-seconds, no paid
compute; CPU setup/tests at most 3600 compute-seconds excluding I/O stalls.
Each fleet smoke <=600s, each awaited stage <=2700s. First use a real-scene
smoke, then a three-setting bounded bandwidth pilot only after correctness.
No training or broad parameter sweep. Preserve each completed stage separately.

Plausible metric bounds before results: finite pixels [0,1] (or uint8 [0,255]);
rendered count 1..selected real trace frame count; CUDA device must be asserted;
payload hashes must match exactly; cached revisit sends zero repeated payloads;
priority only changes delivery scheduling, never payload bytes. No exact published
VMAF/Table-I reproduction without the original identified artifacts/configuration.

Fleet inspection: gpu6 Ada idle; gpu5 Ada has other users' allocations; gpu1/2/4
busy; gpu3 environment probe timed out. Worker health lists all six fresh.
Global status listing timed out after120s through both gpu2 and gpu6; no jobs
have been submitted by this task. Do not replay unknown requests.

Sources: Apple Mail16Oct PR message and23Oct Zhengsi request; sent3Aug message
says Tariq planned an implementation push. Upstream and cd-athena main contain
TCP placeholder. Known forks: EternalSky23, cd-athena, nested mrezaghafari.
Current official moq-lite omits subgroups; modern moq-rs is a functional port,
not the original custom subgroup transport. Keep this distinction explicit.

Delegation: builder reviewed PR and found invalid plane signs, incomplete SH
RGB layout fix, send-versus-delivery subscription state bug. Designer reviewed
official MoQ APIs. Runner stopped on first DNS failure as instructed; native
model/effort telemetry was not exposed, so actual overrides remain unverified.
No remote agent sessions have been launched.

Jobs: none. GPU spend:0/3600s. Results: pending.

## First GPU validation submitted
- Job `20261101T131954Z-e5f936c7`: published/pending, GPU6-only because isolated environment is host-local.
- Deadline `2026-11-01T14:04:31.378643+00:00`; conservative reservation1800/3600GPU-s.
- Source archive SHA256`f524b22acda49871aaf72b96f3f9951af7817f19c890cb66ceb886fce73c63ea`; spec SHA256`c66b013885635ad961064724fcfef1b2ad15b756b0f42459c09df955a26e7289`.
- Source/test/src trees verified equal upstream and server baseline; server uncommitted work untouched.
- Results `/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/requests/20261101T131954Z-e5f936c7` (canonical fleet status determines exact artifact directory).
- Native model completion callback has not been verified in this desktop client; use one bounded fleet wait and retain its handle, no recurring timer.

## Fork audit complete
All5branches across upstream, cd-athena, EternalSky23, and nested mrezaghafari inspected at recorded SHAs; no Rust/native transport artifacts or releases. Server blob691d2aed2ba06bac181bf9ab7e0f0812bf71f116 identical everywhere, explicit TCPplaceholder. Only EternalSky23 client differs. Audit JSONsavedinreproduction/fork-audit-20261101.json.

## Failed first smoke — infrastructure, no numerical evidence
Job20261101T131954Z-e5f936c7 terminalFAILED. CPUregression subprocess timed out180s before output while importing from slowNFS tigasbase. No CUDAimage produced; no fullstage promoted. Savedrun/status duration215.902433s; conservative charge300GPU-s/3600,3300remaining. Failure/source/archive preserved. Revise environment to existing read-only local packed PointStreambase32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661 plus isolated overlay, rather than repeat the sameNFSconfiguration. Newbase PyTorch2.10/CUDA12.8 changesenvironment so all smokes must rerun; do not transfer oldevidence.

## Local-environment combined validation submitted
Job`20261101T134114Z-eeb9ec59`: published/pending. Deadline2026-11-01T14:25:49.004375+00:00;2400smax job,smoke600/full1200. Conservative300GPU-s used;2700/3600reservedincludingnewjob,900unreserved. SourceSHAfd54c2279f7b533847b9241e7f8993da14064116ab75ff7dc4c73827c5ef6bd5; specSHAc291e2bcdc62c1dbc5e6d7f5986a5f104985460deffc58f0ad94edf7ad091f88; adapterSHA96d0173779cbd864efb4db69c950549898b94288b622b1ac92f1302785c30b38. Same combined path bothstages:CPUtests;fullsceneCUDAoracle;legacyTCPclient;3processnativeQUIC. NewbaseTorch2.10+cu128/gsplat1.5.3 inhost-local isolatedoverlay, notoriginalpaperenvironment.

## Second smoke failed safely at CPU regressions
Job20261101T134114Z-eeb9ec59 terminalFAILED,50passed/2failedin7.74s. Execution11.349262s, conservatively charge60GPU-s; aggregate360/3600,3240remaining. No numericalCUDAimage evidence. Onefailure was missingimageioFFMPEGplugin (undeclared dependency); the other newtest overconstrainederrorwording. Addeddeclaredimageio[ffmpeg]dependency, writerinitializationfailfast, correctedtestmessage, and correctedlegacyvalidator todecodeMP4onlyafterflush. ActualCUDAtoolkitmustbeexplicitlyselectedbeforethirdsmoke; CPUtestshowsgsplatfoundnoneondefaultPATH. Newsourcearchivewillbeversionedseparately.

Accounting correction: include secondjob staging/supervisor interval, notonly11.35sexecution. Conservative secondcharge180s; aggregate480/3600GPU-s,3120remaining. Sourcev2addsimageio[ffmpeg], writerfailfast,53CPUtests andexplicitlocalCUDA12.4toolkit forTorchcu128 compatibilitysmoke. Existingjobsourcev1unchanged.

## Third request uncertain — do not replay
ID20261101T135740Z-6f49d789, sourcev2SHA82ba075292c97bd41a2a011129f3223215af67d8b27b72d9122797d0e43b30c3. SubmissionlostconnectionwhiletransferringsnapshottoGPU2:Cannotassignrequestedaddress/Brokenpipe. NativeCLIrecordstatussubmission_unknown. InspectthroughGPU6beforeanynewsubmission; nopromotion/evidencetransferassumed.

## Interrupted transfer resolved without replay
Read-onlynativeartifactexportandGPU6directoryinspectionconfirm20261101T135740Z-6f49d789 isUNPUBLISHED:ready.json,spec.json,state.jsonallabsent;noowner,norun;onlyevents/andpartial131072bytesource.tar. Workeradmissionrequiresready.json,soitcannotexecute. Retainthisnon-runnableorphananditslocalrecord;0GPU-s. The installedsingle-jobstatuspathignoresmanagement-hostenvironmentoverrides, andsnapshottransferusesthefirstconfiguredhost. Usepublic--state-dirwithtask-ownedcopiedverifiedconfiguration,managementandfirsttransferhostGPU6;sameexistingdoctor,release,workers,inboxandadmissiongates,nochangestosharedclientconfigurationorworkers. Newattemptcanproceedonlyafterthisinspection.

## GPU6-management attempt published
Job20261101T144900Z-dc0598d9:published/pending, isolatedclientstate/tmp/moqsplat-fleet-state-20261101 requiredforstatus/wait. Deadline2026-11-01T15:33:39.544272+00:00; budget2400s; priorconservative480/3600GPU-s,2880reserved,720unreserved. ExternalcodearchiveSHA101a053494711c2fc96127f8f902e43b9bf9c90f19d8b7d81316265e2c5300e2;specSHAa87f93b2967aac1ea951000981bf04f356c55d11592a3046169f568174e5f8ae;adapterSHAeb52337c34f6412a2a9acdf53635cc9008e1d30a77e918947633a46a77e4f5b3. Nativewaitmustinclude--state-dir. Newpose-conventionauditinitiated:originalCSVconverterFOVmapping/scenealignmentmustbecheckedfromEyeNavSIBRsource beforeanytrajectoryfidelityclaim; currentGLnumericalsmokealonecannotestablishthis.

## Corrected-camera/header run published
Job20261101T153034Z-1dbe7112:published/pending;state-dir/tmp/moqsplat-fleet-state-20261101. Deadline2026-11-01T16:15:29.168861+00:00;budget2400s. FourthattemptFAILED with62CPUtestspassed andCUDAheadercompilefailure;95.35sexecution/116ssupervisor, conservatively180charged. Total660/3600GPU-sbeforethisrun;3060reservedincludingnewmaximum,540unreserved. SourceSHAe5558e99fda54ce3306b141c253e0f41ad75188d0b090239d133fd6475f69af7;specSHAc294e07e52c690f26d8d642ea73e5dfbe06202253e9e844b21f20f192dedbfe0;adapterSHA4e5c1375f8d2ce65b98e68fee414eececa714b2d513a581a5a31a4da8879c6c5. CorrectedtraceSHA4f3567862ddf4e56ab439a262bd7535980422dd05fe360c1b92b762f3fd97499 fromCSV779a6597c9e9f6fbe2c44533195f0cd43a2aa929b450e8c4ad354472679b1f85, exactGitblobd5bf12bcec984401a52d7b82f7f5a224794d09f9 matchespinnedupstream.

## Fifth native smoke failure and sixth request snapshot correction
20261101T153034Z-1dbe7112:FAILED;76CPUtestspassed,fullGPUCUDAkernelscompiled/launched, thenpinnedpackedgsplat1.5.3rejectednewbackgroundshape(1,3). Removedunsupportedbackgroundparameter;composepremultipliedcolorswithreturnedalphaandaddedregression.98.9044sexecution/109.51ssupervisor;conservative180scharge,aggregate840/3600GPU-s.
20261101T154427Z-9679149a publishedbutrequestedCANCEL:afteradapterbecametracked,theexplicitnarrowedsnapshotomitteditsmodulepath. Thisisownselectionerror,notmodelresult. Preserve/cancelonlythisrequest;addexplicit--snapshot-pathexperiments/audit/moqsplat_reproduction.pybeforeanynextsubmission. Snapshotpathsinspecteddirectly. Sourcev5SHAb21d919d33b33e04ba4e2b54873c507faa764a25682f20b9e5f85e5e1578e4fa. Needterminalconfirmationbeforeanotherattempt.

## Corrected snapshot validation submitted
Job `20261101T160216Z-31d696de`: published; state-dir `/tmp/moqsplat-fleet-state-20261101`. Deadline `2026-11-01T16:47:09.350114+00:00`; maximum2400 GPU-s, conservative prior900/3600 GPU-s,3300 reserved. Source v5 SHA256 `b21d919d33b33e04ba4e2b54873c507faa764a25682f20b9e5f85e5e1578e4fa`; spec `6b59d64596f2bab7441f67526c61ba24d8fb89c17b2f69b5b285f39ffe30d0dd`; fleet snapshot `793af4d68595848369d8b3bc2d99866c03cc953245a41424710e14c47f100ff7`, PointStream71153358. Adapter explicitly included and selective archive import preflight passed. Sixth request9679149a confirmed FAILED before execution (missing module); conservative60s charged, no replay.

## Real-scene renderer and legacy client passed; native group limit failed
Job `20261101T160216Z-31d696de` terminalFAILED. CPU77passed; full6,131,954-Gaussian source CUDA oracle and legacyTCP nonblack MP4 passed. Nativepublisher failed with pinned FFI Protocol(group too large), before ready; no nativeGPU success claimed. Execution81.83s, supervisorapprox95s; conservative180s charge, aggregate1080/3600GPU-s. Preserve images/logs; fix explicit native group capacity without scene reduction before next run.

## Native segmentation full-scene validation submitted
Job `20261101T161843Z-425ac71f`: published/pending, state-dir `/tmp/moqsplat-fleet-state-20261101`; deadline2026-11-01T17:03:27.185868+00:00; maximum2400 GPU-s, aggregateprior1080/3600,3480 reserved. Sourcev6SHA256 `d64052cbee3e821d6dc83fb0fd767f883ec119c358165ce9bb011706a108d2c8`, codecommit42b6bbc, spec044fda1b18c3905c9f328008c9f55c1bb94d81a7e4ba43a58484074c74ea54bb, PointStream71153358 selective snapshot793af4d68595848369d8b3bc2d99866c03cc953245a41424710e14c47f100ff7. Native10regressionspassed; separate33sidle lateQUIC recovery passed. Recheckedall6hosts;gpu6bothidle andsolehostwithqualifiedisolatedenvironment. Useoneboundednativefleetwait, no recurringtimer.
Bounded wait handle44468; output `/tmp/moqsplat-v6-bounded-wait-20261101.json`. No recurring wakeups. All existing jobs inspected before new submission; only sourcev6 was newly admitted.

## Final decision — correctness gate passed
Job20261101T161843Z-425ac71f COMPLETE, smoke/full validatorspassed. CPU82passed; sourceCUDAandlegacyTCPpassedbothscales; native3processQUIC receivedall6,131,954Gaussians,5layers,1,447,148,624payloadbytes,exactsourceattributes. Full3final+3progressiveframes; overlapslive transfers;45changedpriorities/72poses;cache revisitrequests0. Finalimagesvisuallyinspected; directsourcecomparisonmaxabsolute8bitRGBerrors1,2,2 across3frames. Execution100.31ssmoke/87.68sfull; supervisor209.38s, charge360swithstaging, aggregate1440/3600GPU-s. Stopdecisionmade, no bandwidthsweep/training. Terminalevent38106ddc873c517d8c91d3727517672d acknowledged; boundedwaitended204s; no tasktimerscreatedorremaining. Canonicalevidenceinjobinbox;localexports/tmp/moqsplat-v6-validated-20261101 and/tmp/moqsplat-v6-full-images-20261101. NativeFFIlibrarySHAaa313c69b57e287234cd2d925bdeeaa27bdb9609c3bf6c059a9d5700f4b8a36a;FFmpeg7.0.2-staticplugin0.7.0;CUDAcompiler12.4.131. Readablepublicreport reproduction/VALIDATION-20261101.md records original-wire/performance/semantic-publisher limits.
Validation clarification: installed fleet invokes external validator for smoke only. Full command self-checks passed; the identical adapter validator was separately run locally against exported fullsummary and passed after completion. Publicreport distinguishes smoke fleetgate from post-run fullartifact verification; do not describe two fleetpromotiongates.
