function iconRect(position, iconSize) {
  const left = Number(position?.x) || 0;
  const top = Number(position?.y) || 0;
  const width = Math.max(0, Number(iconSize?.width) || 0);
  const height = Math.max(0, Number(iconSize?.height) || 0);
  return { left, top, right: left + width, bottom: top + height };
}

export function rectanglesOverlap(left, right) {
  return left.left < right.right
    && left.right > right.left
    && left.top < right.bottom
    && left.bottom > right.top;
}

export function positionIntersectsReservedRects(position, iconSize, reservedRects = []) {
  const candidate = iconRect(position, iconSize);
  return reservedRects.some((reserved) => rectanglesOverlap(candidate, reserved));
}

export function rowMajorGridPosition(index, {
  grid,
  surfaceWidth,
  iconSize,
  reservedRects = [],
} = {}) {
  const safeIndex = Math.max(0, Math.floor(Number(index) || 0));
  const offset = Number(grid?.offset) || 0;
  const cellW = Math.max(1, Number(grid?.cellW) || 1);
  const cellH = Math.max(1, Number(grid?.cellH) || 1);
  const usableWidth = Math.max(cellW, (Number(surfaceWidth) || 1024) - offset * 2);
  const columns = Math.max(1, Math.floor(usableWidth / cellW));
  let accepted = -1;

  for (let slot = 0; slot < safeIndex + reservedRects.length * columns * 8 + 1; slot += 1) {
    const position = {
      x: offset + (slot % columns) * cellW,
      y: offset + Math.floor(slot / columns) * cellH,
    };
    if (positionIntersectsReservedRects(position, iconSize, reservedRects)) continue;
    accepted += 1;
    if (accepted === safeIndex) return position;
  }

  return {
    x: offset,
    y: offset + (safeIndex + reservedRects.length + 1) * cellH,
  };
}

export function positionOutsideReservedRects(position, {
  iconSize,
  bounds,
  reservedRects = [],
} = {}) {
  const minX = Number(bounds?.minX) || 0;
  const minY = Number(bounds?.minY) || 0;
  const maxX = Math.max(minX, Number(bounds?.maxX) || minX);
  const maxY = Math.max(minY, Number(bounds?.maxY) || minY);
  const width = Math.max(1, Number(iconSize?.width) || 1);
  const height = Math.max(1, Number(iconSize?.height) || 1);
  const clamp = (value, min, max) => Math.max(min, Math.min(Number(value) || min, max));
  const original = {
    x: clamp(position?.x, minX, maxX),
    y: clamp(position?.y, minY, maxY),
  };
  if (!positionIntersectsReservedRects(original, { width, height }, reservedRects)) return original;

  const candidates = [];
  for (const reserved of reservedRects) {
    candidates.push(
      { x: reserved.left - width, y: original.y },
      { x: reserved.right, y: original.y },
      { x: original.x, y: reserved.top - height },
      { x: original.x, y: reserved.bottom },
    );
  }

  const viable = candidates
    .map((candidate) => ({
      x: clamp(candidate.x, minX, maxX),
      y: clamp(candidate.y, minY, maxY),
    }))
    .filter((candidate) => !positionIntersectsReservedRects(candidate, { width, height }, reservedRects))
    .sort((left, right) => (
      Math.hypot(left.x - original.x, left.y - original.y)
      - Math.hypot(right.x - original.x, right.y - original.y)
    ));
  if (viable.length) return viable[0];

  for (let y = minY; y <= maxY; y += height) {
    for (let x = minX; x <= maxX; x += width) {
      const candidate = { x, y };
      if (!positionIntersectsReservedRects(candidate, { width, height }, reservedRects)) return candidate;
    }
  }
  return original;
}
