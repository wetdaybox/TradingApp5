# ── Render Each Bot (with side-by-side actual vs. recommended) ──
for key, label, hist, (pr, ch), prob in [
    ("b", "🟡 BTC/USDT", btc_hist, (btc_p, btc_ch), p_b),
    ("x", "🟣 XRP/BTC",   xrp_hist, (xrp_p, None),    p_x),
]:
    low, up, tp, actual_n, rec_n, act = auto_state(
        key, hist, pr, ch, prob,
        st.session_state.get(f"cont_low_{key}", pr),
        st.session_state.get(f"cont_up_{key}", pr),
        st.session_state.get(f"cont_grids_{key}", GRID_MAX),
    )

    st.subheader(f"{label} Bot")

    # two columns: actual grids and recommended grids
    c1, c2 = st.columns(2)
    if st.session_state.mode == "new":
        c1.metric("Grid Levels",    f"{actual_n}")
        c2.metric("Recommended",    f"{rec_n}")
    else:
        # in Continue mode, only show actual
        c1.metric("Grid Levels",    f"{actual_n}")
        c2.write("")  # keep alignment

    # then the rest of your metrics
    st.metric("Lower Price",    f"{low:,.6f}")
    st.metric("Upper Price",    f"{up:,.6f}")
    st.metric("Take-Profit At", f"{tp:,.6f}")

    # action banner
    if act == "Not Deployed":
        st.info("⚠️ Waiting to deploy when conditions are met.")
    elif act == "Redeploy":
        st.info("🔔 Auto grid reset signal detected.")
    elif act == "Take-Profit":
        st.success("💰 TAKE-PROFIT executed—bot terminated.")
    elif act == "Terminated":
        st.error("🛑 Bot terminated—awaiting regime recovery.")
    else:
        st.info("⏸ HOLD—no action right now.")
