// NGO Dashboard JavaScript - Version 2.0
console.log('✅ NGO Dashboard JS loaded - Version 2.0');

let currentNGO = null;
let notificationInterval = null;

async function loadFood() {
    const email = document.getElementById('ngoEmail').value;
    if (!email) {
        alert('Please enter your email');
        return;
    }

    const alertBox = document.getElementById('alertBox');
    const container = document.getElementById('foodContainer');

    try {
        // Get NGO details
        const ngoResponse = await fetch('/api/user/by-email?email=' + encodeURIComponent(email));
        const ngoData = await ngoResponse.json();

        if (!ngoResponse.ok) {
            alertBox.className = 'alert alert-error';
            alertBox.innerHTML = '❌ NGO not found. Please register first.';
            return;
        }

        currentNGO = ngoData;

        if (currentNGO.user_type !== 'ngo') {
            alertBox.className = 'alert alert-error';
            alertBox.innerHTML = '❌ This email is registered as Donor, not NGO.';
            return;
        }

        // Show notification bell and logout button
        document.getElementById('notificationBell').style.display = 'block';
        document.getElementById('logoutBtn').style.display = 'block';
        document.getElementById('tabsContainer').style.display = 'flex';

        // Start loading notifications
        loadNotifications();

        // Auto-refresh notifications every 30 seconds
        if (notificationInterval) clearInterval(notificationInterval);
        notificationInterval = setInterval(loadNotifications, 30000);

        // Get food using TALUKA parameter (FIXED!)
        const timestamp = new Date().getTime();
        let apiUrl;
        
        if (currentNGO.taluka) {
            apiUrl = '/api/food/available-in-area?taluka=' + encodeURIComponent(currentNGO.taluka) + '&t=' + timestamp;
            console.log('🔍 Using TALUKA parameter:', currentNGO.taluka);
        } else {
            apiUrl = '/api/food/available-in-area?area=' + encodeURIComponent(currentNGO.area) + '&t=' + timestamp;
            console.log('🔍 Using AREA parameter:', currentNGO.area);
        }
        
        console.log('🔍 API URL:', apiUrl);
        
        const foodResponse = await fetch(apiUrl);
        const foodData = await foodResponse.json();
        
        console.log('🔍 API Response:', foodData);
        console.log('🔍 Total foods received:', foodData.foods ? foodData.foods.length : 0);

        if (!foodResponse.ok) {
            alertBox.className = 'alert alert-error';
            alertBox.innerHTML = '❌ Error loading food';
            return;
        }

        alertBox.className = 'alert alert-success';
        alertBox.innerHTML = `✅ Logged in as: ${currentNGO.name} | Showing food from ${currentNGO.taluka || currentNGO.area}`;

        // Display food grouped by TALUKA
        if (foodData.foods.length === 0) {
            container.innerHTML = '<div class="empty">😔 No food available in your area yet</div>';
        } else {
            // Group food by taluka first, then area as fallback
            const grouped = {};
            foodData.foods.forEach(food => {
                const key = food.taluka || food.area || 'Unknown Location';
                if (!grouped[key]) {
                    grouped[key] = [];
                }
                grouped[key].push(food);
            });
            
            console.log('🔍 Grouped foods:', grouped);
            Object.keys(grouped).forEach(location => {
                console.log(`🔍 ${location}: ${grouped[location].length} foods`);
            });

            // Create HTML with grouped sections
            let html = '';
            let globalIndex = 0;
            Object.keys(grouped).sort().forEach(location => {
                html += `
                    <div style="margin-bottom: 40px;">
                        <h2 style="background: linear-gradient(135deg, #4CAF50, #45a049); color: white; padding: 15px 25px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3); display: flex; align-items: center; gap: 10px;">
                            📍 ${location}
                            <span style="background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; font-size: 14px; margin-left: auto;">
                                ${grouped[location].length} items
                            </span>
                        </h2>
                        <div class="food-grid">
                            ${grouped[location].map(food => createFoodCard(food, globalIndex++)).join('')}
                        </div>
                    </div>
                `;
            });

            container.innerHTML = html;
            
            console.log('✅ HTML rendered, total cards should be:', globalIndex);

            // Start countdown timers for all food items
            setTimeout(() => {
                foodData.foods.forEach(food => {
                    updateCountdown(food.id, food.expiry_time);
                });
            }, 100);
        }

    } catch (error) {
        console.error('❌ Error:', error);
        alertBox.className = 'alert alert-error';
        alertBox.innerHTML = '❌ Network error: ' + error.message;
    }
}

console.log('✅ loadFood function defined');
