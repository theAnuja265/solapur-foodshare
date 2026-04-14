// Shared impact scripts: animated counters + subtle particles
(async function(){
  function animateNumberEl(el, value, suffix=''){
    if(!el) return;
    const start = parseInt(el.textContent.replace(/\D/g,'')) || 0;
    const end = parseInt(value || 0, 10);
    const duration = 1200;
    const startTime = performance.now();
    function frame(t){
      const p = Math.min(1, (t - startTime)/duration);
      const eased = 1 - Math.pow(1-p, 2);
      const cur = Math.round(start + (end - start) * eased);
      el.textContent = cur + (cur>0?suffix:'');
      if(p < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  async function fetchAndUpdate(){
    try{
      const res = await fetch('/api/stats');
      if(!res.ok) return;
      const d = await res.json();
      // mapping: id -> value
      const map = {
        impactMeals: d.total_meals || d.total_food_shared || 0,
        impactMealsSaved: d.total_food_shared || 0,
        impactNGOs: d.total_ngos || 0,
        impactNGOsConnected: d.total_ngos || 0,
        impactDonors: d.total_donors || 0,
        impactActiveDonors: d.total_donors || 0,
        impactWastePrevented: d.food_saved_kg || d.food_saved || d.food_available || 0,
        impactBeneficiaries: d.beneficiaries || 0,
        impactResponseTime: d.avg_response_min || 0,
        impactSuccessRate: d.success_rate || 0
      };
      Object.entries(map).forEach(([id,val])=>{
        const el = document.getElementById(id);
        if(el) animateNumberEl(el, val, (id==='impactSuccessRate'?'%': (id==='impactResponseTime'?'':'+' )));
      });
    }catch(e){ console.debug('impact fetch failed', e); }
  }

  // initial run and interval
  document.addEventListener('DOMContentLoaded', ()=>{ fetchAndUpdate(); setInterval(fetchAndUpdate, 30000); });

  // Simple particle layer: inject a few circles to the .impact-particles container
  function seedParticles(){
    const containers = document.querySelectorAll('.impact-particles');
    containers.forEach(c=>{
      if(c.getAttribute('data-seeded')) return;
      const svgNS = 'http://www.w3.org/2000/svg';
      const svg = document.createElementNS(svgNS,'svg');
      svg.setAttribute('preserveAspectRatio','none');
      svg.setAttribute('viewBox','0 0 100 100');
      const colors = ['#3dbd73','#2f9bd8','#ffb248','#9b5dd8','#78d0a8'];
      for(let i=0;i<12;i++){
        const cx = Math.random()*100; const cy = Math.random()*100; const r = 0.8 + Math.random()*3.8;
        const circle = document.createElementNS(svgNS,'circle');
        circle.setAttribute('cx',cx); circle.setAttribute('cy',cy); circle.setAttribute('r',r);
        circle.setAttribute('fill', colors[i % colors.length]);
        circle.setAttribute('class','dot');
        circle.style.opacity = 0.45 + Math.random()*0.4;
        svg.appendChild(circle);
        // animate via CSS transform using random durations
        const animDur = 8 + Math.random()*12;
        circle.style.transition = `transform ${animDur}s ease-in-out infinite`;
        (function(el,d){
          setInterval(()=>{
            const tx = (Math.random()-0.5)*2; const ty = (Math.random()-0.5)*4;
            el.style.transform = `translate(${tx}px, ${ty}px)`;
          }, d*1000);
        })(circle, animDur);
      }
      c.appendChild(svg);
      c.setAttribute('data-seeded','1');
    });
  }
  document.addEventListener('DOMContentLoaded', seedParticles);

  // Background rotator for .impact-bg and .carousel-bg containers with enhanced features
  function seedBgRotation(){
    const containers = document.querySelectorAll('.impact-bg, .carousel-bg');
    containers.forEach(c=>{
      if(c.getAttribute('data-rotor')) return;
      // build two layers for crossfade
      const a = document.createElement('div'); a.className = 'bg-layer';
      const b = document.createElement('div'); b.className = 'bg-layer';
      c.appendChild(a); c.appendChild(b);
      const imagesAttr = c.getAttribute('data-images') || '';
      const images = imagesAttr.split(',').map(s=>s.trim()).filter(Boolean);
      if(images.length === 0) images.push('/static/images/volunteers_food.jpg');

      let idx = 0; let top = 0;
      let isPlaying = true;
      
      // init
      a.style.backgroundImage = `url('${images[0]}')`; a.classList.add('show');
      if(images.length>1) b.style.backgroundImage = `url('${images[1]}')`;

      // Add indicators for carousel-bg
      if(c.classList.contains('carousel-bg') && images.length > 1){
        const indicatorContainer = document.createElement('div');
        indicatorContainer.style.cssText = 'position: absolute; bottom: 25px; left: 50%; transform: translateX(-50%); display: flex; gap: 12px; z-index: 10; pointer-events: auto;';
        images.forEach((_, i) => {
          const dot = document.createElement('div');
          dot.style.cssText = `width: 12px; height: 12px; border-radius: 50%; background: rgba(255,255,255,${i===0?0.9:0.4}); cursor: pointer; transition: all 0.3s ease; border: 2px solid rgba(255,255,255,0.8);`;
          dot.onclick = () => jumpToImage(i);
          dot.classList.add('carousel-indicator');
          dot.setAttribute('data-idx', i);
          indicatorContainer.appendChild(dot);
        });
        c.appendChild(indicatorContainer);
      }

      // Pause on hover
      c.addEventListener('mouseenter', () => { isPlaying = false; });
      c.addEventListener('mouseleave', () => { isPlaying = true; });

      function updateIndicators(){
        const dots = c.querySelectorAll('.impact-indicator, .carousel-indicator');
        dots.forEach(dot => {
          const dotIdx = parseInt(dot.getAttribute('data-idx'));
          if(dotIdx === idx){
            dot.style.background = 'rgba(255,255,255,0.95)';
            dot.style.transform = 'scale(1.4)';
          } else {
            dot.style.background = 'rgba(255,255,255,0.4)';
            dot.style.transform = 'scale(1)';
          }
        });
      }

      function jumpToImage(targetIdx){
        idx = targetIdx;
        const nextIdx = (idx + 1) % images.length;
        const front = c.children[top];
        const back = c.children[1-top];
        back.style.backgroundImage = `url('${images[idx]}')`;
        back.classList.add('show');
        front.classList.remove('show');
        top = 1-top;
        updateIndicators();
      }

      function rotate(){
        if (!isPlaying) return;
        const nextIdx = (idx + 1) % images.length;
        const front = c.children[top];
        const back = c.children[1-top];
        back.style.backgroundImage = `url('${images[nextIdx]}')`;
        back.classList.add('show');
        front.classList.remove('show');
        top = 1-top; idx = nextIdx;
        updateIndicators();
      }

      const interval = parseInt(c.getAttribute('data-interval')) || 5000;
      const handle = setInterval(rotate, interval);
      c.setAttribute('data-rotor','1');
      c._rotor = handle;
      
      // Initial indicator update
      updateIndicators();
    });
  }
  document.addEventListener('DOMContentLoaded', seedBgRotation);
})();
